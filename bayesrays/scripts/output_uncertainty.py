import numpy as np
import torch
import nerfstudio
import pkg_resources
import nerfacc
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from bayesrays.utils.utils import normalize_point_coords, find_grid_indices
from nerfstudio.utils import colors
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss
)



def get_uncertainty(self, points):
    aabb = self.scene_box.aabb.to(points.device)
    ## samples outside aabb will have 0 coeff and hence 0 uncertainty. To avoid problems with these samples we set zero_out=False
    inds, coeffs = find_grid_indices(points, aabb, self.field.spatial_distortion ,self.lod, points.device, zero_out=False)
    cfs_2 = (coeffs**2)/torch.sum((coeffs**2),dim=0, keepdim=True)
    uns = self.un[inds.long()] #[8,N]
    un_points = torch.sqrt(torch.sum((uns*cfs_2),dim=0)).unsqueeze(1)
    
    #for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points+1e-12)
    un_points = un_points.view((points.shape[0], points.shape[1],1))
    return un_points

def get_output_nerfacto_new(self, ray_bundle):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''
#     original_outputs = self.__class__.get_outputs(self, ray_bundle)  # Call original get_outputs (this is slower than just copying the original method here)
    
    N = self.N
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    density_fns_new = []
    if self.filter_out:
        for i in self.density_fns:
            density_fns_new.append(lambda x, i=i: i(x) * (self.get_uncertainty(x)<= self.filter_thresh*max_uncertainty))
    else:
        density_fns_new = self.density_fns
    
    if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    else:
        ray_samples,_, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
    points = ray_samples.frustums.get_positions()
    un_points = self.get_uncertainty(points)

    #get weights
    if self.filter_out:
        density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
    else:
        density = field_outputs[FieldHeadNames.DENSITY]
    weights = ray_samples.get_weights(density)
    
    uncertainty = torch.sum(weights * un_points, dim=-2) 
    uncertainty += (1-torch.sum(weights,dim=-2)) * min_uncertainty #alpha blending
    
    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    
    if self.white_bg:
        self.renderer_rgb.background_color=colors.WHITE 
    elif self.black_bg:
        self.renderer_rgb.background_color=colors.BLACK     
    rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    accumulation = self.renderer_accumulation(weights=weights)

    original_outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
    }
                                    
    original_outputs['uncertainty'] = uncertainty 
    if self.training:
        original_outputs["weights_list"] = weights_list
        original_outputs["ray_samples_list"] = ray_samples_list
    if self.config.predict_normals:
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        original_outputs["normals"] = self.normals_shader(normals)
        original_outputs["pred_normals"] = self.normals_shader(pred_normals)    

    if self.training and self.config.predict_normals:
        original_outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        )

        original_outputs["rendered_pred_normal_loss"] = pred_normal_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS].detach(),
            field_outputs[FieldHeadNames.PRED_NORMALS],
        )

    for i in range(self.config.num_proposal_iterations):
        original_outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])


    return original_outputs

def get_output_ngp_new(self, ray_bundle):
    assert self.field is not None
    assert pkg_resources.get_distribution("nerfstudio").version >= "0.3.1"
    
    N = self.N  #approx ray dataset size
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H =   reg_lambda + self.hessian/N
    self.un = 1/H 
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    if self.filter_out:
        density_fn_new = lambda x: self.field.density_fn(x) * (self.get_uncertainty(x.unsqueeze(0)).squeeze(0)<= self.filter_thresh*max_uncertainty)
        self.sampler.density_fn = density_fn_new    
    num_rays = len(ray_bundle)
    if self.collider is not None:
        ray_bundle = self.collider(ray_bundle)
    with torch.no_grad():
        ray_samples, ray_indices = self.sampler(
            ray_bundle=ray_bundle,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            render_step_size=self.config.render_step_size*0.001,
            alpha_thre=self.config.alpha_thre,
            cone_angle=self.config.cone_angle,
        )

    field_outputs = self.field(ray_samples)
    points = ray_samples.frustums.get_positions()
    un_points = self.get_uncertainty(points.unsqueeze(0)).squeeze(0)
    if self.filter_out:
        density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
    else:
        density = field_outputs[FieldHeadNames.DENSITY] 
    # accumulation
    packed_info = nerfacc.pack_info(ray_indices, num_rays)
    weights = nerfacc.render_weight_from_density(
        t_starts=ray_samples.frustums.starts[..., 0],
        t_ends=ray_samples.frustums.ends[..., 0],
        sigmas=density[..., 0],
        packed_info=packed_info,
    )[0]
        
    weights = weights[..., None]
    
    comp_uncertainty = nerfacc.accumulate_along_rays(
                weights[..., 0], values=un_points, ray_indices=ray_indices, n_rays=num_rays
            )
    accumulated_weight = nerfacc.accumulate_along_rays(
        weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
    )
    uncertainty = comp_uncertainty + min_uncertainty * (1.0 - accumulated_weight)#alpha blending

    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    
    if self.white_bg:
        self.renderer_rgb.background_color=colors.WHITE    
    elif self.black_bg:
        self.renderer_rgb.background_color=colors.BLACK  

    rgb = self.renderer_rgb(
        rgb=field_outputs[FieldHeadNames.RGB],
        weights=weights,
        ray_indices=ray_indices,
        num_rays=num_rays,
    )
    depth = self.renderer_depth(
        weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
    )
    accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
    
    outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
        "uncertainty": uncertainty
    }
    return outputs

def get_output_mipnerf_new(self, ray_bundle):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''
    
    N = self.N  #approx ray dataset size
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    
    # uniform sampling
    ray_samples = self.sampler_uniform(ray_bundle)
        
    # First pass:
    field_outputs_coarse = self.field.forward(ray_samples)
    weights_coarse = ray_samples.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
    rgb_coarse = self.renderer_rgb(
        rgb=field_outputs_coarse[FieldHeadNames.RGB],
        weights=weights_coarse,
    )
    accumulation_coarse = self.renderer_accumulation(weights_coarse)
    depth_coarse = self.renderer_depth(weights_coarse, ray_samples)

    # pdf sampling
    ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples, weights_coarse)

    # Second pass:
    field_outputs_fine = self.field.forward(ray_samples_pdf)
        
    points = ray_samples_pdf.frustums.get_positions()
    un_points = self.get_uncertainty(points)
    
    if self.filter_out:
        density = field_outputs_fine[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
    else:
        density = field_outputs_fine[FieldHeadNames.DENSITY] 
    #get weights
    weights_fine = ray_samples_pdf.get_weights(density)
    rgb_fine = self.renderer_rgb(
        rgb=field_outputs_fine[FieldHeadNames.RGB],
        weights=weights_fine,
    )
    accumulation_fine = self.renderer_accumulation(weights_fine)
    depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)
    
    
    uncertainty = torch.sum(weights_fine * un_points, dim=-2) 
    uncertainty += (1-torch.sum(weights_fine,dim=-2)) * min_uncertainty #alpha blending
    
    #normalize into acceptable range for rendering
    uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)
    
    if self.white_bg:
        self.renderer_rgb.background_color=colors.WHITE    
    elif self.black_bg:
        self.renderer_rgb.background_color=colors.BLACK        
   
    original_outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
            "depth": depth_fine,
            "rgb": rgb_fine,
            "accumulation": accumulation_fine
        }
    original_outputs['uncertainty'] = uncertainty 

    return original_outputs

def get_output_fn(model):

    if isinstance(model, NerfactoModel):
        return get_output_nerfacto_new
    elif isinstance(model, NGPModel):
        return get_output_ngp_new
    elif isinstance(model, MipNerfModel):
        return get_output_mipnerf_new
    else:
        raise Exception("Sorry, this model is not currently supported.")

def get_output_nerfacto_all(self, ray_bundle):
    ''' reimplementation of get_output function from models for evaluation with different filter levels'''
    
    N = self.N  #approx ray dataset size
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(64^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    original_outputs={}
    for thresh in self.thresh_range:
        density_fns_new = []
        for i in self.density_fns:
            density_fns_new.append(lambda x, i=i: i(x) * (self.get_uncertainty(x)<= thresh*max_uncertainty))

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
        else:
            ray_samples,_, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        points = ray_samples.frustums.get_positions()

        #get weights
        uns = self.get_uncertainty(ray_samples.frustums.get_positions())
        density = field_outputs[FieldHeadNames.DENSITY] * (uns <= thresh*max_uncertainty)
        weights = ray_samples.get_weights(density)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        original_outputs["rgb-"+"{:.2f}".format(thresh.item())] = rgb
        original_outputs["accumulation-"+"{:.2f}".format(thresh.item())] = accumulation     
        original_outputs["depth-"+"{:.2f}".format(thresh.item())] = depth     

    return original_outputs

