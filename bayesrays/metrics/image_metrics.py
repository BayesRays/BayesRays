# source: https://github.com/nerfstudio-project/nerfstudio/blob/nerfbusters-changes/nerfstudio/utils/metrics.py

"""
This file contains NeRF metrics with masking capabilities.
"""

from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure
from torchtyping import TensorType
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def _valid_img(img: Tensor, normalize: bool):
    """check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check


class LearnedPerceptualImagePatchSimilarityWithMasking(LearnedPerceptualImagePatchSimilarity):
    """LearnedPerceptualImagePatchSimilarity module that will allow for masking capabilities."""

    def update(self, img1: Tensor, img2: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update internal states with lpips score with masking."""

        # hardcode this to True for now to avoid touching a lot of the torchmetrics code
        self.net.spatial = True

        if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
            )
        loss = self.net(img1, img2, normalize=self.normalize)
        # now loss is the shape [batch size, H, W]
        # we set loss to self.sum_scores to use the existing API from torchvision
        self.sum_scores = loss  # pylint: disable=attribute-defined-outside-init

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        # note that we don't use self.reduction anymore
        return self.sum_scores
    
class ImageMetricModule(nn.Module):
    """Computes image metrics with masking capabilities.
    We assume that the pred and target inputs are in the range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.populate_modules()

    def populate_modules(self):
        """Populates the modules that will be used to compute the metric."""

    @abstractmethod
    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        """Computes the metric.
        Args:
            preds: Predictions.
            target: Ground truth.
            mask: Mask to use to only compute the metrics where the mask is True.
        Returns:
            Metric value.
        """


class PSNRModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        preds_reshaped = preds.view(bs, 3, hw)
        target_reshaped = target.view(bs, 3, hw)
        num = (preds_reshaped - target_reshaped) ** 2
        # the non-masked version
        if mask is None:
            den = hw
        else:
            mask_reshaped = mask.view(bs, 1, hw)
            num = num * mask_reshaped
            den = mask_reshaped.sum(-1)
        mse = num.sum(-1) / den
        psnr = 10 * torch.log10(1.0 / mse)
        psnr = psnr.mean(-1)
        return psnr


class SSIMModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        _, ssim_image = structural_similarity_index_measure(
            preds=preds, target=target, reduction="none", data_range=1.0, return_full_image=True
        )
        ssim_image = ssim_image.mean(1)  # average over the channels
        assert ssim_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            ssim = ssim_image.view(bs, hw).mean(1)
            return ssim

        # the masked version
        ssim_reshaped = ssim_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        ssim = (ssim_reshaped * mask_reshaped / den).sum(-1)
        return ssim


class LPIPSModule(ImageMetricModule):
    """Computes LPIPS with masking capabilities."""

    def populate_modules(self):
        # by setting normalize=True, we assume that the pred and target inputs are in the range [0, 1]
        self.lpips_with_masking = LearnedPerceptualImagePatchSimilarityWithMasking(normalize=True)

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        with torch.no_grad():
            lpips_image = self.lpips_with_masking(preds, target)
        lpips_image = lpips_image.mean(1)  # average over the channels
        assert lpips_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            lpips = lpips_image.view(bs, hw).mean(1)
            return lpips

        # the masked version
        lpips_reshaped = lpips_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        lpips = (lpips_reshaped * mask_reshaped / den).sum(-1)
        return lpips