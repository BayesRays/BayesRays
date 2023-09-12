# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# for older versions of nerfstudio filtering slider is currently not supported.
"""
Starts viewer in eval mode.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import tyro
import torch
import numpy as np
import types

import nerfstudio
import pkg_resources
import nerfacc

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer, colors
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.field_components.field_heads import FieldHeadNames
from bayesrays.utils.utils import find_grid_indices
import torch.nn.functional as F
from bayesrays.scripts.output_uncertainty import get_uncertainty

if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
    from nerfstudio.viewer.server.viewer_state import ViewerState
    from nerfstudio.viewer.server.viewer_elements import  ViewerSlider
else:
    from nerfstudio.viewer.server import viewer_utils
    from nerfstudio.utils.writer import EventName, TimeWriter


def get_output_nerfacto_new(self, ray_bundle):
    ''' reimplementation of get_output function from models because of lack of proper interface to outputs dict'''
    
    N = self.N 
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
    density_fns_new = []
    num_fns = len(self.density_fns) 
    for i in self.density_fns:
        density_fns_new.append(lambda x, i=i: i(x) * (self.get_uncertainty(x)<= self.filter_thresh*max_uncertainty))
        
    if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    else:
        ray_samples,_, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=density_fns_new)
    field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
    points = ray_samples.frustums.get_positions()
    un_points = self.get_uncertainty(points)
    #get weights
    density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
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
    
    
    # this is based on https://arxiv.org/pdf/2211.12656.pdf and the summation is not normalized. Check normalized in the viewer.
    # Uniform sampler (weights_list[0]) is used for getting uniform samples on each ray frustrum and find sum of entropy
    # of ray termination probabilities. Change sum to average for a more correct form. 
    ww = torch.clamp(weights_list[0], 1e-10 ,1.)
    entropy = -torch.sum(ww* torch.log2(ww) + (1-ww) * torch.log2(1-ww), dim=1)
   
    original_outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
    }
    original_outputs['uncertainty'] = uncertainty 
    original_outputs['entropy'] = entropy

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
    
    N = self.N
    reg_lambda = 1e-4 /( (2**self.lod)**3)
    H = self.hessian/N + reg_lambda
    self.un = 1/H 
            
    max_uncertainty = 6 #approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3 #approximate lower bound of that function (cutting off at hessian = 1000)
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
    density = field_outputs[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
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
    
    N = self.N
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
    
    density = field_outputs_fine[FieldHeadNames.DENSITY] * (un_points <= self.filter_thresh*max_uncertainty)
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

        
@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewerU:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    unc_path: Path = Path("unc.npy")
    """Path to output video file."""
    white_bg: bool = True
    """ Render empty space as white when filtering""" 
    black_bg: bool = False
    """ Render empty space as black when filtering""" 

    def main(self) -> None:
        """Main function."""
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, _, step = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
            )
        else:
            config, pipeline, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
            )
            step = 0
        
        self.device = pipeline.device
        pipeline.model.filter_thresh = 1.
        pipeline.model.hessian = torch.tensor(np.load(str(self.unc_path)), device=self.device)
        pipeline.model.N =  4096*1000  #approx ray dataset size (train batch size x number of query iterations in uncertainty extraction step)
        pipeline.model.lod = np.log2(round(pipeline.model.hessian.shape[0]**(1/3))-1)
        pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, pipeline.model)
        pipeline.model.white_bg = self.white_bg
        pipeline.model.black_bg = self.black_bg
        new_method = get_output_fn(pipeline.model)
        pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)
        
        
        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = "viewer"
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk
    
        self._start_viewer(config, pipeline, step)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """
    def _update_viewer_state(self, viewer_state: viewer_utils.ViewerState, pipeline: Pipeline):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        """
        # NOTE: step must be > 0 otherwise the rendering would not happen
        step = 1
        num_rays_per_batch = pipeline.datamanager.get_train_rays_per_batch()
        with TimeWriter(writer, EventName.ITER_VIS_TIME) as _:
            viewer_state.update_scene = types.MethodType(update_scene_new, viewer_state)
            try:
                viewer_state.update_scene(self, step, pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert viewer_state.vis is not None
                viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    def _start_viewer(self, config: TrainerConfig, pipeline: Pipeline, step: int):
        """Starts the viewer

        Args:
            config: Configuration of pipeline to load
            pipeline: Pipeline instance of which to load weights
            step: Step at which the pipeline was saved
        """
        base_dir = config.get_base_dir()
        viewer_log_path = base_dir / config.viewer.relative_log_filename

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            viewer_state = ViewerState(
            config.viewer,
            log_filename=viewer_log_path,
            datapath=pipeline.datamanager.get_datapath(),
            pipeline=pipeline,
            )
            viewer_state.control_panel._filter = ViewerSlider(
                    "Filter Threshold",
                    default_value=1.,
                    min_value=0.0,
                    max_value=1,
                    step=0.05,
                    hint="Filtering threshold for uncertain areas.",
                )
            viewer_state.control_panel.add_element(viewer_state.control_panel._filter)
            banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]
        else:
            viewer_state, banner_messages = viewer_utils.setup_viewer(
                config.viewer, log_filename=viewer_log_path, datapath=pipeline.datamanager.get_datapath()
            )

        # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
        config.logging.local_writer.enable = False
        writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

        assert viewer_state and pipeline.datamanager.train_dataset

        nerfstudio_version = pkg_resources.get_distribution("nerfstudio").version

        if nerfstudio_version >= "0.3.1":
            if nerfstudio_version >=  "0.3.3":
                viewer_state.init_scene(
                    train_dataset=pipeline.datamanager.train_dataset,
                    train_state="completed",
                )
            else:
                viewer_state.init_scene(
                    dataset=pipeline.datamanager.train_dataset,
                    train_state="completed",
                )
            viewer_state.viser_server.set_training_state("completed")
            viewer_state.update_scene(step=step)
            while True:
                time.sleep(0.01)
                pipeline.model.filter_thresh = viewer_state.control_panel._filter.value
        else:
            viewer_state.init_scene(
                dataset=pipeline.datamanager.train_dataset,
                start_train=False,
            )   
            while True:
                viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(viewer_state, pipeline)

                
# this function is redefined to allow support of filter threshold slider in the viewer for old nerfstudio versions.
# The "Train Util." slider in eval time will control the filter threshold instead. 
def update_scene_new(self, trainer, step: int, graph: Model, num_rays_per_batch: int) -> None:
    """updates the scene based on the graph weights

    Args:
        step: iteration step of training
        graph: the current checkpoint of the model
    """
    has_temporal_distortion = getattr(graph, "temporal_distortion", None) is not None
    self.vis["model/has_temporal_distortion"].write(str(has_temporal_distortion).lower())

    is_training = self.vis["renderingState/isTraining"].read()
    self.step = step

    self._check_camera_path_payload(trainer, step)
    self._check_populate_paths_payload(trainer, step)

    camera_object = self._get_camera_object()
    if camera_object is None:
        return

    if is_training is None or is_training:
        # in training mode

        if self.camera_moving:
            # if the camera is moving, then we pause training and update camera continuously

            while self.camera_moving:
                self._render_image_in_viewer(camera_object, graph, is_training)
                camera_object = self._get_camera_object()
        else:
            # if the camera is not moving, then we approximate how many training steps need to be taken
            # to render at a FPS defined by self.static_fps.

            if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                target_train_util = self.vis["renderingState/targetTrainUtil"].read()
                if target_train_util is None:
                    target_train_util = 0.9

                batches_per_sec = train_rays_per_sec / num_rays_per_batch

                num_steps = max(int(1 / self.static_fps * batches_per_sec), 1)
            else:
                num_steps = 1

            if step % num_steps == 0:
                self._render_image_in_viewer(camera_object, graph, is_training)

    else:
        # in pause training mode, enter render loop with set graph
        local_step = step
        run_loop = not is_training
        while run_loop:
            # if self._is_render_step(local_step) and step > 0:
            if step > 0:
                self._render_image_in_viewer(camera_object, graph, is_training)
                camera_object = self._get_camera_object()
            th =  self.vis["renderingState/targetTrainUtil"].read() 
            graph.filter_thresh = th if th is not None else 1.    
            is_training = self.vis["renderingState/isTraining"].read()
            self._check_populate_paths_payload(trainer, step)
            self._check_camera_path_payload(trainer, step)
            run_loop = not is_training
            local_step += 1
                

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RunViewerU).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RunViewerU)  # noqa