from transformers.models.deformable_detr.modeling_deformable_detr import DeformableDetrMSDeformAttn
import torch
import torch.nn.functional as F

class MSDeformAttnFunction:
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
        """
        Forward pass for multi-scale deformable attention using Hugging Face's Transformers.

        Args:
            value: Tensor of shape (N, S, M, D) - input values.
            value_spatial_shapes: Tensor of shape (L, 2) - spatial shapes per level.
            value_level_start_index: Tensor of shape (L,) - start indices for each level.
            sampling_locations: Tensor of shape (N, Lq, M, L, P, 2) - sampling locations.
            attention_weights: Tensor of shape (N, Lq, M, L, P) - attention weights.

        Returns:
            output: Tensor of shape (N, Lq, M, D).
        """
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        output = DeformableDetrMSDeformAttn.apply(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for multi-scale deformable attention.

        Args:
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Gradients with respect to inputs: value, spatial shapes, level start indices, sampling locations, and attention weights.
        """
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = DeformableDetrMSDeformAttn.backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    A reference implementation of multi-scale deformable attention in PyTorch.

    Args:
        value: Tensor of shape (N, S, M, D) - input values.
        value_spatial_shapes: Tensor of shape (L, 2) - spatial shapes per level.
        sampling_locations: Tensor of shape (N, Lq, M, L, P, 2) - sampling locations.
        attention_weights: Tensor of shape (N, Lq, M, L, P) - attention weights.

    Returns:
        output: Tensor of shape (N, Lq, M, D).
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(N_, M_ * D_, Lq_)

    return output.transpose(1, 2).contiguous()
