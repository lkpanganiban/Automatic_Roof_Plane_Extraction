import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from typing import Optional

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            print("Warning: You'd better set d_model to make the dimension of each attention head "
                  "a power of 2 which is more efficient in the CUDA implementation.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * torch.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_head)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}'
            )

        output = self.deformable_attention(
            value, sampling_locations, attention_weights,
            input_spatial_shapes, input_level_start_index, Len_q
        )
        
        return self.output_proj(output)

    def deformable_attention(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        Len_q: int,
    ) -> torch.Tensor:
        N = value.shape[0]
        final_output = torch.zeros((N, Len_q, self.d_model), device=value.device)
        
        value_list = []
        for lvl in range(self.n_levels):
            h, w = spatial_shapes[lvl]
            value_l = value[:, level_start_index[lvl]:level_start_index[lvl] + h * w, :, :]
            value_list.append(value_l)

        sampling_grids = 2 * sampling_locations - 1

        for lvl in range(self.n_levels):
            # Get value and reshape
            h, w = spatial_shapes[lvl]
            value_l = value_list[lvl]
            value_l = value_l.reshape(N, h * w, self.n_heads, self.d_head)
            value_l = value_l.permute(0, 2, 3, 1).reshape(N * self.n_heads, self.d_head, h, w)

            # Get sampling grid for this level
            sampling_grid_l = sampling_grids[:, :, :, lvl, :, :]
            sampling_grid_l = sampling_grid_l.reshape(N * self.n_heads, Len_q, self.n_points, 2)

            # Sample the values
            sampled_l = F.grid_sample(
                value_l, 
                sampling_grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # (N * n_heads, d_head, Len_q, n_points)

            # Reshape sampled values and weights
            sampled_l = sampled_l.reshape(N, self.n_heads, self.d_head, Len_q, self.n_points)
            weights_l = attention_weights[:, :, :, lvl, :]  # (N, Len_q, n_heads, n_points)
            weights_l = weights_l.permute(0, 2, 1, 3)  # (N, n_heads, Len_q, n_points)

            # Apply attention weights
            output_l = sampled_l * weights_l.unsqueeze(2)  # Broadcasting over d_head
            output_l = output_l.sum(dim=-1)  # Sum over points
            
            # Add to final output
            final_output += output_l.permute(0, 2, 1, 3).reshape(N, Len_q, self.d_model)

        return final_output