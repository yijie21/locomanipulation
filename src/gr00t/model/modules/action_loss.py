"""Masked flow loss with optional emphasis on the executable action prefix."""
import math
import torch
import torch.nn.functional as F


def masked_action_loss(prediction, target, mask, prefix_frames=10, prefix_weight=1.0):
    if prefix_frames < 1 or not math.isfinite(prefix_weight) or prefix_weight <= 0:
        raise ValueError('positive prefix length and finite positive weight required')
    # Preserve exactly the historical operation sequence when disabled.
    effective_mask = mask
    if prefix_weight != 1.0:
        weights = torch.ones((1, prediction.shape[1], 1), device=mask.device, dtype=mask.dtype)
        weights[:, :prefix_frames] = prefix_weight
        effective_mask = mask * weights
    per_action = F.mse_loss(prediction, target, reduction='none') * effective_mask
    loss = per_action.sum() / (effective_mask.sum() + 1e-6)
    return loss, per_action, effective_mask
