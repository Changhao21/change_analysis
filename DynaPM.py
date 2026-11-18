"""A PyTorch training-wrapper schematic for DynaPM."""


from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SegmentationCriterion = Callable[[Tensor, Tensor], Tensor]

class DynaPM(nn.Module):


    def __init__(
        self,
        segmenter: nn.Module,
        criterion: Optional[SegmentationCriterion] = None,
        ignore_index: Optional[int] = -100,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.segmenter = segmenter
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if criterion is None:
            if ignore_index is None:
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.criterion = criterion

    @staticmethod
    def _extract_logits(output: object) -> Tensor:
        """Adapt common segmentation-model output containers to a logit tensor."""
        if isinstance(output, Tensor):
            logits = output
        elif isinstance(output, dict):
            logits = None
            for key in ("out", "logits", "seg_logits"):
                value = output.get(key)
                if isinstance(value, Tensor):
                    logits = value
                    break
            if logits is None:
                raise TypeError(
                    "Could not find logits in model output dict; expected one "
                    "of: 'out', 'logits', 'seg_logits'."
                )
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            logits = output[0]
        else:
            raise TypeError(
                "The segmenter must return a Tensor, a supported dict, or a "
                "tuple/list beginning with a Tensor of logits."
            )

        if logits.ndim != 4:
            raise ValueError(
                f"Expected logits with shape [B, K, H, W], got {tuple(logits.shape)}."
            )
        return logits

    def _predict(self, images: Tensor, output_size: Tuple[int, int]) -> Tensor:
        logits = self._extract_logits(self.segmenter(images))
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return logits

    def forward(
        self, images: Tensor, targets: Optional[Tensor] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}.")

        # Inference adds no DynaPM branch: one ordinary segmentation pass.
        if targets is None:
            return self._predict(images, output_size=images.shape[-2:])

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 3:
            raise ValueError(
                f"Expected targets [B, H, W], got {tuple(targets.shape)}."
            )
        if targets.shape[0] != images.shape[0]:
            raise ValueError("Image and target batch sizes must match.")
        if targets.shape[-2:] != images.shape[-2:]:
            raise ValueError("Image and target spatial dimensions must match.")
        if targets.device != images.device:
            raise ValueError("Images and targets must be on the same device.")

        targets = targets.long()
        spatial_size = targets.shape[-2:]
        main_logits = self._predict(images, output_size=spatial_size)

        if main_logits.shape[0] != targets.shape[0]:
            raise ValueError("Segmenter logits and targets must have matching batch sizes.")

        valid = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = targets != self.ignore_index

        # An entirely ignored batch has no supervised pixels; return a
        # graph-connected zero instead of allowing mean-reduced CE to become NaN.
        if not valid.any():
            zero = main_logits.sum() * 0.0
            empty_mask = torch.zeros_like(valid)
            zero_ratio = zero.detach()
            zero_threshold = zero.detach()
            return {
                "loss": zero,
                "loss_main": zero,
                "loss_mask": zero,
                "main_logits": main_logits,
                "masked_logits": main_logits,
                "mask": empty_mask,
                "mask_ratio": zero_ratio,
                "threshold": zero_threshold,
            }

        loss_main = self.criterion(main_logits, targets)
        if not isinstance(loss_main, Tensor) or loss_main.ndim != 0:
            raise ValueError("criterion must return a scalar Tensor.")

        # PDE and MGPD: use category-index deviation and the batch mean as T.
        # Detaching makes the discrete selection independent of backpropagation.
        with torch.no_grad():
            predicted_ids = main_logits.detach().argmax(dim=1)
            deviation = (predicted_ids - targets).abs()
            threshold = deviation[valid].to(dtype=main_logits.dtype).mean()
            mask = (deviation < threshold) & valid
            mask_ratio = mask.sum().to(dtype=main_logits.dtype) / mask.numel()

        # MGSS: preserve selected image pixels and zero all other locations.
        masked_images = images * mask.unsqueeze(1).to(dtype=images.dtype)
        masked_logits = self._predict(masked_images, output_size=spatial_size)
        loss_mask = self.criterion(masked_logits, targets)
        if not isinstance(loss_mask, Tensor) or loss_mask.ndim != 0:
            raise ValueError("criterion must return a scalar Tensor.")

        # ALC: the mined-pixel ratio controls the auxiliary loss contribution.
        total_loss = loss_main + mask_ratio * loss_mask
        return {
            "loss": total_loss,
            "loss_main": loss_main,
            "loss_mask": loss_mask,
            "main_logits": main_logits,
            "masked_logits": masked_logits,
            "mask": mask,
            "mask_ratio": mask_ratio,
            "threshold": threshold,
        }


__all__ = ["DynaPM"]
