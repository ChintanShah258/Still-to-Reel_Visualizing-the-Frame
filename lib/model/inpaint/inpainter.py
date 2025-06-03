# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from kornia.morphology import opening, erosion
from kornia.filters import gaussian_blur2d
from lib.model.inpaint.networks.inpainting_nets import Inpaint_Depth_Net, Inpaint_Color_Net
from lib.utils.render_utils import masked_median_blur


def refine_near_depth_discontinuity(depth, alpha, kernel_size=11):
    '''
    Refines depth discontinuity boundaries using adaptive smoothing and frequency-aware adjustments.
    '''
    depth = depth * alpha
    depth_median_blurred = masked_median_blur(depth, alpha, kernel_size=kernel_size) * alpha
    assert not torch.isnan(depth_median_blurred).any(), "Depth median blur contains NaN values!"
    assert not torch.isinf(depth_median_blurred).any(), "Depth median blur contains Inf values!"
    alpha_eroded = erosion(alpha, kernel=torch.ones(kernel_size, kernel_size).to(alpha.device))

    # Apply hybrid smoothing for depth refinement
    depth_median_blurred = gaussian_blur2d(depth_median_blurred, (5, 5), (1.2, 1.2))
    depth_preserved = torch.where(
        torch.abs(depth - depth_median_blurred) < 0.05,
        depth_median_blurred,  # Smooth small deviations
        depth  # Preserve larger edges
    )
    assert not torch.isnan(depth_preserved).any(), "Depth preserved contains NaN values!"
    assert not torch.isinf(depth_preserved).any(), "Depth preserved contains Inf values!"
    
    depth[alpha_eroded == 0] = depth_preserved[alpha_eroded == 0]
    return depth


def define_inpainting_bbox(alpha, border=40):
    '''
    Define the bounding box for inpainting with adaptive border size.
    '''
    assert alpha.ndim == 4 and alpha.shape[:2] == (1, 1)
    x, y = torch.nonzero(alpha)[:, -2:].T
    h, w = alpha.shape[-2:]
    row_min, row_max = x.min(), x.max()
    col_min, col_max = y.min(), y.max()
    out = torch.zeros_like(alpha)
    # Dynamically adjust the border size based on region size
    border = max(border, 20) if (row_max - row_min) * (col_max - col_min) > 10000 else border
    x0, x1 = max(row_min - border, 0), min(row_max + border, h - 1)
    y0, y1 = max(col_min - border, 0), min(col_max + border, w - 1)
    out[:, :, x0:x1, y0:y1] = 1
    return out


class Inpainter():
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading depth model...")
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load('ckpts/depth-model.pth', map_location=torch.device(device))
        depth_feat_model.load_state_dict(depth_feat_weight)
        depth_feat_model = depth_feat_model.to(device)
        depth_feat_model.eval()
        self.depth_feat_model = depth_feat_model.to(device)
        print("Loading RGB model...")
        rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load('ckpts/color-model.pth', map_location=torch.device(device))
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        self.rgb_model = rgb_model.to(device)

        # Kernels for erosion and opening
        self.context_erosion_kernel = torch.ones(10, 10).to(self.device)
        self.alpha_kernel = torch.ones(3, 3).to(self.device)

    @staticmethod
    def process_depth_for_network(depth, context, log_depth=True):
        if log_depth:
            log_depth = torch.log(depth + 1e-8) * context
            mean_depth = torch.mean(log_depth[context > 0])
            zero_mean_depth = (log_depth - mean_depth) * context
        else:
            zero_mean_depth = depth
            mean_depth = 0
        return zero_mean_depth, mean_depth

    @staticmethod
    def deprocess_depth(zero_mean_depth, mean_depth, log_depth=True):
        if log_depth:
            depth = torch.exp(zero_mean_depth + mean_depth)
        else:
            depth = zero_mean_depth
        return depth

    def inpaint_rgb(self, holes, context, context_rgb, edge):
        '''
        Perform RGB inpainting with residual sharpness enhancement and bilateral filtering.
        '''
        with torch.no_grad():
            inpainted_rgb = self.rgb_model.forward_3P(
                holes, context, context_rgb, edge, unit_length=128, cuda=self.device)
        inpainted_rgb = inpainted_rgb.detach() * holes + context_rgb
        assert not torch.isnan(inpainted_rgb).any(), "Inpainted RGB contains NaN values after model inference!"
        assert not torch.isinf(inpainted_rgb).any(), "Inpainted RGB contains Inf values after model inference!"
        
        # Apply bilateral filtering for perceptual quality
        inpainted_rgb = gaussian_blur2d(inpainted_rgb, (3, 3), (1.0, 1.0))

        # Residual enhancement for sharper output
        residual = context_rgb - inpainted_rgb
        inpainted_rgb += 0.15 * residual  # Amplify residual signal

        assert not torch.isnan(inpainted_rgb).any(), "Inpainted RGB contains NaN values after residual enhancement!"
        assert not torch.isinf(inpainted_rgb).any(), "Inpainted RGB contains Inf values after residual enhancement!"
        
        inpainted_a = opening(holes + context, self.alpha_kernel)
        inpainted_rgba = torch.cat([inpainted_rgb, inpainted_a], dim=1)
        return inpainted_rgba

    def inpaint_depth(self, depth, holes, context, edge, depth_range):
        '''
        Perform depth inpainting with clipping, edge-aware smoothing, and consistency constraints.
        '''
        zero_mean_depth, mean_depth = self.process_depth_for_network(depth, context)
        assert not torch.isnan(zero_mean_depth).any(), "Zero-mean depth contains NaN values!"
        assert not torch.isinf(zero_mean_depth).any(), "Zero-mean depth contains Inf values!"
        with torch.no_grad():
            inpainted_depth = self.depth_feat_model.forward_3P(
                holes, context, zero_mean_depth, edge, unit_length=128, cuda=self.device)
        inpainted_depth = self.deprocess_depth(inpainted_depth.detach(), mean_depth)
        assert not torch.isnan(inpainted_depth).any(), "Deprocessed depth contains NaN values!"
        assert not torch.isinf(inpainted_depth).any(), "Deprocessed depth contains Inf values!"
        
        # Hybrid smoothing and depth clipping
        inpainted_depth = gaussian_blur2d(inpainted_depth, (3, 3), (1.2, 1.2))
        assert not torch.isnan(inpainted_depth).any(), "Inpainted depth after Gaussian blur contains NaN values!"
        assert not torch.isinf(inpainted_depth).any(), "Inpainted depth after Gaussian blur contains Inf values!"
        inpainted_depth = torch.clamp(inpainted_depth, min=min(depth_range) * 0.95, max=max(depth_range) * 1.05)
        return inpainted_depth

    def sequential_inpainting(self, rgb, depth, depth_bins):
        '''
        Sequentially inpaint layers based on depth bins with improved depth and RGB handling.
        '''
        num_bins = len(depth_bins) - 1

        rgba_layers = []
        depth_layers = []
        mask_layers = []

        for i in range(num_bins):
            alpha_i = (depth >= depth_bins[i]) * (depth < depth_bins[i + 1])
            alpha_i = alpha_i.float()

            if i == 0:
                rgba_i = torch.cat([rgb * alpha_i, alpha_i], dim=1)
                rgba_layers.append(rgba_i)
                depth_i = refine_near_depth_discontinuity(depth, alpha_i)
                assert not torch.isnan(depth_i).any(), f"Depth layer {i} contains NaN values!"
                assert not torch.isinf(depth_i).any(), f"Depth layer {i} contains Inf values!"
                depth_layers.append(depth_i)
                mask_layers.append(alpha_i)
                pre_alpha = alpha_i.bool()
                pre_inpainted_depth = depth * alpha_i
            else:
                alpha_i_eroded = erosion(alpha_i, torch.ones(10 + i, 10 + i).to(self.device))
                if alpha_i_eroded.sum() < 10:
                    continue

                context = erosion((depth >= depth_bins[i]).float(), self.context_erosion_kernel)

                holes = 1. - context
                bbox = define_inpainting_bbox(context, border=40)
                holes *= bbox
                edge = torch.zeros_like(holes)
                context_rgb = rgb * context

                # Inpaint depth
                inpainted_depth_i = self.inpaint_depth(depth, holes, context, edge, (depth_bins[i], depth_bins[i + 1]))
                assert not torch.isnan(inpainted_depth_i).any(), f"Inpainted depth for layer {i} contains NaN values!"
                assert not torch.isinf(inpainted_depth_i).any(), f"Inpainted depth for layer {i} contains Inf values!"
                depth_near_mask = (inpainted_depth_i < depth_bins[i + 1]).float()

                # Inpaint RGB
                inpainted_rgba_i = self.inpaint_rgb(holes, context, context_rgb, edge)
                assert not torch.isnan(inpainted_rgba_i).any(), f"Inpainted RGBA for layer {i} contains NaN values!"
                assert not torch.isinf(inpainted_rgba_i).any(), f"Inpainted RGBA for layer {i} contains Inf values!"
                
                if i < num_bins - 1:
                    inpainted_rgba_i *= depth_near_mask
                    inpainted_depth_i = refine_near_depth_discontinuity(inpainted_depth_i, inpainted_rgba_i[:, [-1]])

                inpainted_alpha_i = inpainted_rgba_i[:, [-1]].bool()
                mask_wrong_ordering = (inpainted_depth_i <= pre_inpainted_depth) * inpainted_alpha_i
                inpainted_depth_i[mask_wrong_ordering] = pre_inpainted_depth[mask_wrong_ordering] * 1.05

                rgba_layers.append(inpainted_rgba_i)
                depth_layers.append(inpainted_depth_i)
                mask_layers.append(context * depth_near_mask)

                pre_alpha[inpainted_alpha_i] = True
                pre_inpainted_depth[inpainted_alpha_i > 0] = inpainted_depth_i[inpainted_alpha_i > 0]

        rgba_layers = torch.stack(rgba_layers)
        depth_layers = torch.stack(depth_layers)
        mask_layers = torch.stack(mask_layers)

        return rgba_layers, depth_layers, mask_layers
