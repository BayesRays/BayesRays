
#!/usr/bin/env python
"""
uncertainty.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import types
import time

import tyro
import torch
import numpy as np
import nerfstudio
import nerfacc
import pkg_resources
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.encodings import HashEncoding
from bayesrays.utils.utils import normalize_point_coords, find_grid_indices, get_gaussian_blob_new


@dataclass
class ComputeUncertainty:
    """Load a checkpoint, compute uncertainty, and save it to a npy file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("unc.npy")
    # Uncertainty level of detail (log2 of it)
    lod: int = 8
    # number of iterations on the trainset    
    iters: int = 1000         
    
    def find_uncertainty(self, points, deform_points, rgb, distortion):
        inds, coeffs = find_grid_indices(points, self.aabb, distortion, self.lod, self.device)
        #because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        r = deform_points.grad.clone().detach().view(-1,3)
        deform_points.grad.zero_()
        colors[1].backward(retain_graph=True)
        g = deform_points.grad.clone().detach().view(-1,3)
        deform_points.grad.zero_()
        colors[2].backward()
        b = deform_points.grad.clone().detach().view(-1,3)
        deform_points.grad.zero_()
        dmy = (torch.arange(points.shape[0])[...,None]).repeat((1,points.shape[1])).flatten().to(self.device)
        first = True
        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1),inds[corner].unsqueeze(-1)), dim=-1) 
                all_r = coeffs[corner].unsqueeze(-1)*r
                all_g = coeffs[corner].unsqueeze(-1)*g
                all_b = coeffs[corner].unsqueeze(-1)*b
                first = False
            else:
                all_ind = torch.cat((all_ind, torch.cat((dmy.unsqueeze(-1),inds[corner].unsqueeze(-1)), dim=-1)), dim=0)
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1)*r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1)*g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1)*b ), dim=0)
        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[...,0]) #for first element of deformation field
        grad_g_1 = torch.bincount(inds_all, weights=all_g[...,0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[...,0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[...,1]) #for second element of deformation field
        grad_g_2 = torch.bincount(inds_all, weights=all_g[...,1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[...,1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[...,2]) #for third element of deformation field
        grad_g_3 = torch.bincount(inds_all, weights=all_g[...,2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[...,2])
        grad_1 = grad_r_1**2+grad_g_1**2+grad_b_1**2
        grad_2 = grad_r_2**2+grad_g_2**2+grad_b_2**2
        grad_3 = grad_r_3**2+grad_g_3**2+grad_b_3**2 #will consider the trace of each submatrix for each deformation
        #vector as indicator of hessian wrt the whole vector
        
        grads_all = torch.cat((keys_all[:,1].unsqueeze(-1), (grad_1+grad_2+grad_3).unsqueeze(-1)), dim=-1)
        hessian = torch.zeros(((2**self.lod)+1)**3).to(self.device)
        hessian = hessian.put((grads_all[:,0]).long(), grads_all[:,1], True)

        return hessian
    
    
    def get_unc_nerfacto(self, ray_bundle, model):
        ''' reimplementation of get_output function from models because of lack of proper interface to ray_samples'''
        if model.collider is not None:
            ray_bundle = model.collider(ray_bundle)
        
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":  
            ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
        else:
            ray_samples,_, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
        points = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points, self.aabb, model.field.spatial_distortion)
        #find offset value in the deformation field
        offsets = self.deform_field(pos).clone().detach()
        offsets.requires_grad = True
        
        ray_samples.frustums.set_offsets(offsets)    
        
        
        field_outputs = model.field(ray_samples, compute_normals=model.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = model.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if model.config.predict_normals:
            normals = model.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = model.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = model.normals_shader(normals)
            outputs["pred_normals"] = model.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if model.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if model.training and model.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(model.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = model.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs, points, offsets
    
    
    def get_unc_ngp(self, ray_bundle, model):
        assert model.field is not None
        assert pkg_resources.get_distribution("nerfstudio").version >= "0.3.1"
        
        num_rays = len(ray_bundle)
        if model.collider is not None:
            ray_bundle = model.collider(ray_bundle)
        with torch.no_grad():
            ray_samples, ray_indices = model.sampler(
                ray_bundle=ray_bundle,
                near_plane=model.config.near_plane,
                far_plane=model.config.far_plane,
                render_step_size=model.config.render_step_size*0.001,
                alpha_thre=model.config.alpha_thre,
                cone_angle=model.config.cone_angle,
            )
        points = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points, self.aabb, model.field.spatial_distortion)
        #find offset value in the deformation field
        offsets = self.deform_field(pos).clone().detach()
        offsets.requires_grad = True
        
        ray_samples.frustums.set_offsets(offsets) 
        
        field_outputs = model.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = model.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = model.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = model.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        return outputs, points.unsqueeze(0), offsets

    
    
    def get_unc_mipnerf(self, ray_bundle, model):
        ''' reimplementation of get_output function from models because of lack of proper interface to ray_samples'''
        if model.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")
            
        
        if model.collider is not None:
            ray_bundle = model.collider(ray_bundle)
        
        
        
        # uniform sampling
        ray_samples = model.sampler_uniform(ray_bundle)

        points_coarse = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points_coarse, self.aabb, model.field.spatial_distortion)
        #find offset value in the deformation field
        offsets_coarse = self.deform_field(pos).clone().detach()
        offsets_coarse.requires_grad = True
        
        ray_samples.frustums.set_offsets(offsets_coarse) 
        ray_samples.frustums.get_gaussian_blob = types.MethodType(get_gaussian_blob_new, ray_samples.frustums)


        # First pass:
        field_outputs_coarse = model.field.forward(ray_samples)
        weights_coarse = ray_samples.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = model.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = model.renderer_accumulation(weights_coarse)
        depth_coarse = model.renderer_depth(weights_coarse, ray_samples)

        # pdf sampling
        ray_samples_pdf = model.sampler_pdf(ray_bundle, ray_samples, weights_coarse)
        
        points_fine = ray_samples_pdf.frustums.get_positions()
        pos, _ = normalize_point_coords(points_fine, self.aabb, model.field.spatial_distortion)
        #find offset value in the deformation field
        offsets_fine = self.deform_field(pos).clone().detach()
        offsets_fine.requires_grad = True
        
        ray_samples_pdf.frustums.set_offsets(offsets_fine) 
        ray_samples_pdf.frustums.get_gaussian_blob = types.MethodType(get_gaussian_blob_new, ray_samples_pdf.frustums)

        # Second pass:
        field_outputs_fine = model.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = model.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = model.renderer_accumulation(weights_fine)
        depth_fine = model.renderer_depth(weights_fine, ray_samples_pdf)
        
        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs, points_fine, offsets_fine, points_coarse, offsets_coarse
    
    def get_output_fn(self, model):
        
        if isinstance(model, NerfactoModel):
            return self.get_unc_nerfacto
        elif isinstance(model, NGPModel):
            return self.get_unc_ngp
        elif isinstance(model, MipNerfModel):
            return self.get_unc_mipnerf
        else:
            raise Exception("Sorry, this model is not currently supported.")
            
    def main(self) -> None:
        """Main function."""
        
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        else:
            config, pipeline, checkpoint_path = eval_setup(self.load_config)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        
        self.device = pipeline.device
        self.aabb = pipeline.model.scene_box.aabb.to(self.device)
        self.hessian = torch.zeros(((2**self.lod)+1)**3).to(self.device)
        self.deform_field = HashEncoding(num_levels = 1, 
                            min_res = 2**self.lod,
                            max_res = 2**self.lod,
                            log2_hashmap_size = self.lod*3+1, #simple regular grid (hash table size > grid size)
                            features_per_level = 3,
                            hash_init_scale = 0.,
                            implementation = "torch",
                            interpolation = "Linear")
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2**self.lod]).to(self.device)
        
        pipeline.eval()
        len_train = max(pipeline.datamanager.train_dataset.__len__(), self.iters)
        for step in range(len_train):
            print("step",step)
            ray_bundle, batch = pipeline.datamanager.next_train(step)
            output_fn = self.get_output_fn(pipeline.model)
            if not isinstance(pipeline.model, MipNerfModel):
                outputs, points, offsets = output_fn(ray_bundle, pipeline.model)
                hessian = self.find_uncertainty(points, offsets, outputs['rgb'], pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
            else:
                outputs, points_fine, offsets_fine, points_coarse, offsets_coarse = output_fn(ray_bundle, pipeline.model)
                hessian = self.find_uncertainty(points_fine, offsets_fine, outputs['rgb_fine'], pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
                hessian = self.find_uncertainty(points_coarse, offsets_coarse, outputs['rgb_coarse'], pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
        end_time = time.time()    
        print("Done")
        with open(str(self.output_path), 'wb') as f:
            np.save(f, self.hessian.cpu().numpy())
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")    


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeUncertainty).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeUncertainty)  # noqa
