
""" Same as nerfstudio dataparse but uses CF-NeRF and NerfingMVS train/test splits for LF and ScanNet datasets """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import RandomState


MAX_AUTO_RESOLUTION = 1600
seed = 12345


@dataclass
class SparseNsDataParserConfig(DataParserConfig):
    """Sparse Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: SparseNerfstudio)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = 2
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    dataset_name: Literal["africa", "basket", "statue", "torch", "scene_079","scene_000","scene_158","scene_316"] = "africa"


@dataclass
class SparseNerfstudio(DataParser):
    """Sparse Nerfstudio DatasetParser"""

    config: SparseNsDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)
        
        if "mask_path" in frame:
            sorted_lists = sorted(zip(image_filenames, mask_filenames))
            _, mask_filenames = zip(*sorted_lists)
        if "depth_file_path" in frame:
            sorted_lists = sorted(zip(image_filenames, depth_filenames))
            _, depth_filenames = zip(*sorted_lists)    
            
        sorted_lists = sorted(zip(image_filenames, poses))
        image_filenames, poses = zip(*sorted_lists)
        
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(image_filenames)
            
            i_all = np.arange(num_images)
            
#train/test split from CF-NeRF https://github.com/poetrywanderer/CFNeRF/blob/66918a9748c137e1c0242c12be7aa6efa39ece06/run_nerf_uncertainty_NF.py#L750
            if self.config.dataset_name == 'basket':
                # 4 views
                i_train = list(np.arange(43,50,2))
                i_val = list(np.arange(42,50,2))
                self.config.scene_scale = 50. # half the far bound computed by colmap bounds 
            elif self.config.dataset_name == 'africa':
                # 5 views
                i_train = list(np.arange(5,14,2))
                i_val = list(np.arange(6,14,2))
                self.config.scene_scale = 36.

            elif self.config.dataset_name == 'statue':
                # 5 views
                i_train = list(np.arange(67,76,2))
                i_val = list(np.arange(68,76,2))
                self.config.scene_scale = 50.

            elif self.config.dataset_name == 'torch':
                # 5 views
                i_train = list(np.arange(8,17,2))
                i_val = list(np.arange(9,17,2))
                self.config.scene_scale = 72.
                
# train/test split from scannet scenes of https://github.com/weiyithu/NerfingMVS/tree/28511191239daf25cd8ded17e7fa21a68df54de1
            elif self.config.dataset_name == 'scene_079':
                i_val = [4, 12, 20, 28, 37]
                i_train = list([i for i in range(40) if i not in i_val])
                self.config.scene_scale = 150.
            elif self.config.dataset_name == 'scene_000':
                i_val = [4, 12, 20, 28, 36]
                i_train = list([i for i in range(40) if i not in i_val])
                self.config.scene_scale = 140.
            elif self.config.dataset_name == 'scene_316':
                i_val = [4, 12, 20, 28, 36]
                i_train = list([i for i in range(40) if i not in i_val])
                self.config.scene_scale = 100.
            elif self.config.dataset_name == 'scene_158':
                i_val = [4, 12, 20, 28, 36]
                i_train = list([i for i in range(40) if i not in i_val]) 
                self.config.scene_scale = 2000
            
            else:
                raise ValueError(f"Unknown scene {self.config.dataset_name}")
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_val
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method,
        )
        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
