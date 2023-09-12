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
"""
eval.py
"""
from __future__ import annotations

from datetime import datetime
import json
import types
import torch
import mediapy as media
from dataclasses import dataclass
from time import time
from pathlib import Path
from typing import Optional
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
import tyro
import numpy as np
import cv2

import nerfstudio
import pkg_resources

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import colormaps

from bayesrays.metrics.ause import ause
from bayesrays.metrics.image_metrics import PSNRModule, SSIMModule, LPIPSModule
from torchmetrics import MeanSquaredError


from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import inferno

from bayesrays.scripts.output_uncertainty import get_output_nerfacto_new, get_output_mipnerf_new, get_output_ngp_new, get_output_fn, get_output_nerfacto_all, get_uncertainty

def plot_errors(ratio_removed, ause_err, ause_err_by_var, err_type, scene_no, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, ause_err, '--')
    plt.plot(ratio_removed, ause_err_by_var, '-r')
    # plt.plot(ratio_removed, ause_err_by_var - ause_err, '-g') # uncomment for getting plots similar to the paper, without visible oracle curve
    path = output_path.parent / Path("plots") 
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path/ Path('plot_'+err_type+'_'+str(scene_no)+'.png'))
    plt.figure()

def visualize_ranks(unc ,gt, colormap='jet'):
    flattened_unc = unc.flatten()
    flattened_gt = gt.flatten()
    
    # Find the ranks of the pixel values
    ranks_unc = np.argsort(np.argsort(flattened_unc)) 
    ranks_gt = np.argsort(np.argsort(flattened_gt)) 
    
    max_rank = max(np.max(ranks_unc),np.max(ranks_gt))
    
    cmap = plt.get_cmap(colormap, max_rank)
    
    # Normalize the ranks to the range [0, 1]
    normalized_ranks_unc = ranks_unc / max_rank
    normalized_ranks_gt = ranks_gt / max_rank
    
    # Apply the colormap to the normalized ranks
    colored_ranks_unc = cmap(normalized_ranks_unc)
    colored_ranks_gt = cmap(normalized_ranks_gt)
    
    colored_unc = colored_ranks_unc.reshape((*unc.shape,4))
    colored_gt = colored_ranks_gt.reshape((*gt.shape,4))
    
    return colored_unc, colored_gt

def get_filtered_image_metrics(self, 
                                 outputs: Dict[str, torch.Tensor],
                                 batch: Dict[str, torch.Tensor], thresh: torch.Tensor, add_nb_mask=False, visibility_mask: torch.Tensor=None) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    
    image = batch["image"]
    rgb = outputs["rgb-"+"{:.2f}".format(thresh.item())]
    h,w = rgb.shape[0], rgb.shape[1]
    image = torch.clip(torch.tensor(media.resize_image(image, (h, w))).to(self.device),0.,1.)
    
    acc = outputs["accumulation-"+"{:.2f}".format(thresh.item())]
    depth = outputs["depth-"+"{:.2f}".format(thresh.item())]
    #implementing masked psnr,lpisp,ssim like https://github.com/ethanweber
    #/nerfbusters/blob/1f4240344ecff1313f6dfa7be5e06fe7d3e29154/scripts/launch_nerf.py#L258
    depth_mask = (depth < 2.0).float()
   
    if add_nb_mask:
        mask = depth_mask[...,0] * visibility_mask
        mask = mask[..., None].repeat(1, 1, 3)
    else:
        mask = depth_mask[...,0]
        mask = mask[..., None].repeat(1, 1, 3)
    image, rgb = image * mask, rgb * mask
    
    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    image = image.permute(2, 0, 1).unsqueeze(0)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    m = mask.permute(2, 0, 1).unsqueeze(0)[:, 0:1]

    psnr = float(self.psnr_module(rgb, image, m)[0])
    ssim = float(self.ssim_module(rgb, image, m)[0])
    lpips = float(self.lpips_module(rgb, image, m)[0])

    
    metrics_dict = {"psnr": float(psnr), "ssim": float(ssim)}  # type: ignore
    metrics_dict["lpips"] = float(lpips)
    if add_nb_mask:
        metrics_dict["coverage"] = float(mask[..., 0].sum() / visibility_mask.sum() * 100)
    else:    
        metrics_dict["coverage"] = float(mask[..., 0].sum()/(image.shape[-1]*image.shape[-2]) * 100)
    return metrics_dict

def get_average_filtered_image_metrics(self, step: Optional[int] = None):
    """Iterate over all the images in the eval dataset and get the average.
    From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py#L342

    Returns:
        metrics_dict: dictionary of metrics
    """
    self.eval()
    num_images = len(self.datamanager.fixed_indices_eval_dataloader)

    # Override evaluation function
    self.model.get_image_metrics_and_images = types.MethodType(get_filtered_image_metrics, self.model)
    
    thresh_values = torch.linspace(0.1,1,10)
    self.model.thresh_range = thresh_values
    
    self.model.psnr_module = PSNRModule().to(self.device)
    self.model.ssim_module = SSIMModule().to(self.device)
    self.model.lpips_module = LPIPSModule().to(self.device)
    
    views = ["view"]
    psnr = [["psnr-"+"{:.2f}".format(i.item())] for i in thresh_values]
    lpips = [["lpisp-"+"{:.2f}".format(i.item())] for i in thresh_values]
    ssim = [["ssim-"+"{:.2f}".format(i.item())] for i in thresh_values]
    coverage = [["coverage-"+"{:.2f}".format(i.item())] for i in thresh_values]
    self.datamanager.fixed_indices_eval_dataloader.cameras.rescale_output_resolution(1./self.downscale_factor)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        view_no = 0 
        
        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            
            if self.add_nb_mask:
                base_path = self.nb_mask_path
                pseudo_gt_visibility = media.read_image(str(base_path)+"/{:05d}.png".format(view_no))
                pseudo_gt_visibility = torch.from_numpy(pseudo_gt_visibility).long().to(self.device)
                pseudo_gt_visibility = (pseudo_gt_visibility[..., 0] >= 1).float()
            else:
                pseudo_gt_visibility = 1
            
            # time this the following line
            inner_start = time()
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            for n,thresh in enumerate(thresh_values):
                metrics_dict = self.model.get_image_metrics_and_images(outputs, batch, thresh, self.add_nb_mask, pseudo_gt_visibility)
                psnr[n].append(float(metrics_dict["psnr"]))
                lpips[n].append(float(metrics_dict["lpips"]))
                ssim[n].append(float(metrics_dict["ssim"]))
                coverage[n].append(float(metrics_dict["coverage"]))
                
                print("view:", view_no, "-", str(thresh), "psnr:", psnr[n][-1], "coverage:", coverage[n][-1])

            views.append(str(view_no))    
            view_no +=1
            progress.advance(task)
            

    # average the metrics list
    metrics_dict = {}

    lists = [*psnr, *ssim, *lpips, *coverage]
    for l in lists:
        l.append(sum(l[1:])/view_no)
    views.append('average')
    lists = [views] + lists 
    self.train()
    return {}, lists



def get_image_metrics_and_images_unc(self, no:int,
                                 outputs: Dict[str, torch.Tensor],
                                 batch: Dict[str, torch.Tensor], 
                                 err_all: list,
                                 err_var_all: list,
                                 last: bool,
                                 eval_depth: bool) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """ From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py#L357 """
    image = batch["image"].to(self.device)
    rgb = outputs["rgb"]

    unc = outputs["uncertainty"]

    acc = colormaps.apply_colormap(outputs["accumulation"])
    depth = colormaps.apply_depth_colormap(
        outputs["depth"],
        accumulation=outputs["accumulation"],
    )
    combined_rgb = torch.cat([image, rgb], dim=1)
    combined_acc = torch.cat([acc], dim=1)
    combined_depth = torch.cat([depth], dim=1)
    
    if eval_depth:
        depth = outputs["depth"].squeeze(-1)
        
        # load the calculated scale, to run evaluation on depth in same scale as GT depth
        a = float(np.loadtxt(str(self.dataset_path) +'/scale_parameters.txt', delimiter=','))
        depth = a * depth 
        depth_gt_dir = str(self.dataset_path) + '/depth_gt_{:02d}.npy'.format(no)
        depth_gt = np.load(depth_gt_dir)
        depth_gt = torch.tensor(depth_gt, device=depth.device)
        depth = depth/depth_gt.max()
        depth_gt = depth_gt/depth_gt.max()


        squared_error = ((depth_gt - depth) ** 2)
        absolute_error = (abs(depth_gt - depth))
        unc_flat = unc.flatten()
        absolute_error_flat = absolute_error.flatten()
        squared_error_flat = squared_error.flatten()

        ratio, err_mse, err_var_mse, ause_mse = ause(unc_flat, squared_error_flat, err_type='mse')
        plot_errors(ratio, err_mse, err_var_mse, 'mse', no, self.output_path)
        ratio, err_mae, err_var_mae, ause_mae =  ause(unc_flat, absolute_error_flat, err_type='mae')
        plot_errors(ratio, err_mae, err_var_mae, 'mae', no, self.output_path)
        ratio, err_rmse, err_var_rmse, ause_rmse =  ause(unc_flat, squared_error_flat, err_type='rmse')
        plot_errors(ratio, err_rmse, err_var_rmse, 'rmse', no, self.output_path)

        err_all[0] += err_mse
        err_all[1] += err_rmse
        err_all[2] += err_mae
        err_var_all[0] += err_var_mse
        err_var_all[1] += err_var_rmse
        err_var_all[2] += err_var_mae

        if last:
            ratio_all = ratio
            err_mse_all, err_rmse_all, err_mae_all = err_all[0]/(no+1), err_all[1]/(no+1), err_all[2]/(no+1)
            err_var_mse_all, err_var_rmse_all, err_var_mae_all = err_var_all[0]/(no+1), err_var_all[1]/(no+1), err_var_all[2]/(no+1)
            plot_errors(ratio_all, err_mse_all, err_var_mse_all, 'mse', 'all', self.output_path)
            plot_errors(ratio_all, err_rmse_all, err_var_rmse_all, 'rmse', 'all', self.output_path)
            plot_errors(ratio_all, err_mae_all, err_var_mae_all, 'mae', 'all', self.output_path)



        #for visualizaiton
        depth_img = torch.clip(depth, min=0., max=1.)
        absolute_error_img = torch.clip(absolute_error, min=0., max=1.)
    
    #save images
    path = self.output_path.parent / "plots" 
    path.mkdir(parents=True, exist_ok=True)
    if eval_depth:
        im = Image.fromarray((depth_gt.cpu().numpy()* 255).astype('uint8'))
        im.save(path / Path(str(no)+"_depth_gt.jpeg"))
        im = Image.fromarray((depth_img.cpu().numpy()* 255).astype('uint8'))
        im.save(path / Path(str(no)+"_depth.jpeg"))
        im = Image.fromarray(np.uint8(inferno(absolute_error_img.cpu().numpy()) * 255) )
        im.save(path / Path(str(no)+"_error.png"))

        im = Image.fromarray((image.cpu().numpy()* 255).astype('uint8'))
        im.save(path / Path(str(no)+"_gt_image.jpeg"))
        uu, errr = visualize_ranks(unc.squeeze(-1).cpu().numpy(), absolute_error.cpu().numpy())
        im = Image.fromarray(np.uint8(uu * 255))
        im.save(path / Path(str(no)+"_unc_colored.png"))

        im = Image.fromarray(np.uint8(errr * 255))
        im.save(path / Path(str(no)+"_error_colored.png"))

    
  
    im = Image.fromarray(np.uint8(inferno(unc.squeeze(-1).cpu().numpy()) * 255).astype('uint8'))
    im.save(path / Path(str(no)+"_unc.png"))
    im = Image.fromarray(np.uint8(rgb.cpu().numpy() * 255).astype('uint8'))
    im.save(path / Path(str(no)+"_rgb.jpeg"))
    
    
    
    # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
    image = torch.moveaxis(image, -1, 0)[None, ...]
    rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

    psnr = self.psnr(image, rgb)
    ssim = self.ssim(image, rgb)
    lpips = self.lpips(image, rgb)


    # all of these metrics will be logged as scalars
    metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
    metrics_dict["lpips"] = float(lpips)
    if eval_depth:
        metrics_dict["ause_mse"] = float(ause_mse)
        metrics_dict["ause_mae"] = float(ause_mae)
        metrics_dict["ause_rmse"] = float(ause_rmse)
        metrics_dict["mse"] = float(squared_error.mean().item())
    
    images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
    if eval_depth:
        images_dict['err_all'] = err_all
        images_dict['err_var_all'] = err_var_all


    return metrics_dict, images_dict


def get_average_uncertainty_metrics(self, step: Optional[int] = None):
    """Iterate over all the images in the eval dataset and get the average.
    From https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/pipelines/base_pipeline.py#L342

    Returns:
        metrics_dict: dictionary of metrics
    """
    self.eval()
    metrics_dict_list = []
    unc_list = []
    mse_list = []
    num_images = len(self.datamanager.fixed_indices_eval_dataloader)

    # Override evaluation function
    self.model.get_image_metrics_and_images = types.MethodType(get_image_metrics_and_images_unc, self.model)
    
    views = ["view"]
    psnr = ["psnr"]
    lpips = ["lpips"]
    ssim = ["ssim"]
    ause_mse = ["ause_mse"]
    ause_rmse = ["ause_rmse"]
    ause_mae = ["ause_mae"]
    mse = ["mse"]
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        view_no = 0
        err_all = [np.zeros(100),np.zeros(100),np.zeros(100)]
        err_var_all = [np.zeros(100),np.zeros(100),np.zeros(100)] 

        for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
            
            # time this the following line
            inner_start = time()
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(view_no, outputs, batch, err_all, err_var_all, view_no == len(self.datamanager.fixed_indices_eval_dataloader)-1, self.eval_depth)
            if self.eval_depth:
                mse_list.append(metrics_dict["mse"])
                err_all = images_dict['err_all'] 
                err_var_all = images_dict['err_var_all']

            
            # TODO do this in a cleaner way
            views.append(str(view_no))
            psnr.append(float(metrics_dict["psnr"]))
            lpips.append(float(metrics_dict["lpips"]))
            ssim.append(float(metrics_dict["ssim"]))
            if self.eval_depth:
                ause_mse.append(float(metrics_dict["ause_mse"]))
                ause_mae.append(float(metrics_dict["ause_mae"]))
                ause_rmse.append(float(metrics_dict["ause_rmse"]))
                mse.append(float(metrics_dict["mse"]))

            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
            metrics_dict_list.append(metrics_dict)
            view_no +=1
            progress.advance(task)
            
    # average the metrics list
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metrics_dict[key] = float(
            torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
        )
    
    if self.eval_depth:    
        lists = [psnr, ssim, lpips, ause_mse, ause_rmse, ause_mae, mse]
    else:
        lists = [psnr, ssim, lpips]
    for l in lists:
        l.append(sum(l[1:])/view_no)
    views.append('average')
   
    lists = [views] + lists
    self.train()
    return metrics_dict, lists


@dataclass
class ComputeMetrics:
    """Load a checkpoint, compute some metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Path to output video file.
    unc_path: Path = Path("unc.npy")
    # Render with filtering.
    dataset_path: Path = Path("./data")
    # dataset path    
    downscale_factor: float = 2.0
    filter_out: bool = False
    # filter floater results
    nb_mask: bool = False
    # add extra visibility mask to metrics like nerfbusters for fair evaluation
    visibility_path: Path = Path("./data/nerfbusters/aloe/visibility_masks")
    # if nb_mask set to true, specify the path to the masks
    eval_depth: bool = True
    #perform evaluation on depth error
        

    def main(self) -> None:
        """Main function."""
        
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        else:
            config, pipeline, checkpoint_path = eval_setup(self.load_config)
        
        # Dynamic change of get_outputs method to include uncertainty
        self.device = pipeline.device
        pipeline.model.filter_out = self.filter_out
        pipeline.model.hessian = torch.tensor(np.load(str(self.unc_path))).to(self.device)
        pipeline.model.lod = np.log2(round(pipeline.model.hessian.shape[0]**(1/3))-1)
        pipeline.model.dataset_path = self.dataset_path
        pipeline.model.output_path = self.output_path
        pipeline.model.white_bg = False
        pipeline.model.black_bg = False
        pipeline.model.N =  4096*1000 #approx ray dataset size (train batch size x number of query iterations in uncertainty extraction step)
        pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, pipeline.model)
        pipeline.expname = config.experiment_name
        pipeline.add_nb_mask = self.nb_mask
        pipeline.nb_mask_path = self.visibility_path
        pipeline.downscale_factor = self.downscale_factor
        pipeline.eval_depth = self.eval_depth
        if not self.filter_out:
            new_method = get_output_fn(pipeline.model)
            pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)
        else: #filter out metrics only implemented for nerfacto currently!
            pipeline.model.get_outputs = types.MethodType(get_output_nerfacto_all, pipeline.model)

        # Override evaluation function
        if self.filter_out:
            pipeline.get_average_eval_image_metrics = types.MethodType(get_average_filtered_image_metrics, pipeline)
        else:    
            pipeline.get_average_eval_image_metrics = types.MethodType(get_average_uncertainty_metrics, pipeline)

        assert self.output_path.suffix == ".json"
        metrics_dict, metric_lists = pipeline.get_average_eval_image_metrics()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        
            
        # Save output to output file
        nb_filter = '_nb' if self.nb_mask else ''
        timestamp = datetime.now().timestamp()
        date_time = datetime.fromtimestamp(timestamp)
        str_date_time = date_time.strftime("%d-%m-%Y-%H%M%S")
        csv_path = str(self.output_path).split('.')[0] + '_' + config.experiment_name + '_'+ str_date_time + nb_filter +'.csv'
        
        np.savetxt(csv_path, [p for p in zip(*metric_lists)], delimiter=',', fmt='%s')
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeMetrics).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs


def get_parser_fn(): return tyro.extras.get_parser(ComputeMetrics)  # noqa
