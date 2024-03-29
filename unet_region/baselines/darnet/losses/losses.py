from __future__ import division, print_function

import torch
import torch.nn as nn

from unet_region.baselines.darnet.utils.contour_utils import evolve_active_rays_fast


class DistanceLossFast(nn.Module):
    def __init__(self, delta_t=2e-4, max_steps=200):
        super(DistanceLossFast, self).__init__()
        self.delta_t = delta_t
        self.max_steps = max_steps

    def forward(self,
                rho_init,
                rho_target,
                origin,
                beta,
                data,
                kappa,
                theta,
                delta_theta,
                debug_hist=False):
        result = evolve_active_rays_fast(
            rho_init,
            beta.squeeze(),
            data.squeeze(),
            kappa.squeeze(),
            theta,
            delta_theta,
            rho_target,
            origin,
            delta_t=self.delta_t,
            max_steps=self.max_steps,
            debug_hist=debug_hist)

        if debug_hist:
            rho, rho_hist = result
        else:
            rho = result

        rho_diff = torch.nn.functional.l1_loss(rho, rho_target.squeeze())

        with torch.no_grad():
            _, height, width = data.squeeze().size()
            # Join two tensors of (N, L) to (N, L, 2)
            # Then add the origin (N, 1, 2) to broadcast
            rho_cos_theta = rho * torch.cos(theta)
            rho_sin_theta = rho * torch.sin(theta)
            joined = torch.stack([rho_cos_theta, rho_sin_theta], dim=2)
            origin = torch.repeat_interleave(origin.unsqueeze(1), rho.shape[1], 1)
            contour =  origin + joined

            out = {'rho_diff': rho_diff,
                   'contour_pred': contour,
                   'rho': rho}

            if debug_hist:
                batch_size, hist_len, contour_len = rho_hist.size()
                rho_cos_theta = rho_target * torch.cos(theta.squeeze())
                rho_sin_theta = rho_target * torch.sin(theta.squeeze())
                joined = torch.stack([rho_cos_theta, rho_sin_theta], dim=2)
                contour_target = origin + joined
                rho_target_x = contour_target[..., 0]
                rho_target_y = contour_target[..., 1]

                # rho_hist is (N, H, L)
                # rho_hist_cos_theta is (N, H, L)
                # joined is (N, H, L, 2)
                rho_hist_cos_theta = rho_hist * torch.cos(theta).unsqueeze(1)
                rho_hist_sin_theta = rho_hist * torch.sin(theta).unsqueeze(1)
                rho_hist_joined = torch.stack(
                    [rho_hist_cos_theta, rho_hist_sin_theta], dim=3)
                rho_hist_xy = rho_hist_joined + origin.unsqueeze(1).unsqueeze(
                    2)
                rho_hist_x = rho_hist_xy[..., 0]
                rho_hist_y = rho_hist_xy[..., 1]

                out['rho_hist_x'] = rho_hist_x
                out['rho_hist_y'] = rho_hist_y
                out['rho_target_x'] = rho_target_x
                out['rho_target_y'] = rho_target_y

        return out
