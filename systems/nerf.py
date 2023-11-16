import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR

# ROBNERF_PATCH_N = 64
# ROBNERF_PATCH_N = 32
ROBNERF_PATCH_N = 16
ROBNERF_PATCH_SIZE = 16
ROBNERF_PATCH_SIZE_2 = ROBNERF_PATCH_SIZE * ROBNERF_PATCH_SIZE
ROBNERF_PATCH_SIZE_HALF = int(ROBNERF_PATCH_SIZE / 2)
ROBNERF_INNER_PATCH_SIZE = 8

@systems.register('nerf-system')
class NeRFSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }

        if "robust_loss" in self.config.system.loss and self.config.system.loss.robust_loss:
            self.train_num_rays = ROBNERF_PATCH_N * ROBNERF_PATCH_SIZE_2
        else:
            self.train_num_rays = self.config.model.train_num_rays

        self.train_num_samples = self.train_num_rays * self.config.model.num_samples_per_ray

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        elif "robust_loss" in self.config.system.loss and self.config.system.loss.robust_loss:
            index_ = torch.randint(0, len(self.dataset.all_images), size=(ROBNERF_PATCH_N,), device=self.dataset.all_images.device)
            index = torch.empty(ROBNERF_PATCH_N * ROBNERF_PATCH_SIZE_2, dtype=index_.dtype, device=index_.device)
            for ind in range(ROBNERF_PATCH_N):
                index[ind * ROBNERF_PATCH_SIZE_2: (ind + 1) * ROBNERF_PATCH_SIZE_2] = index_[ind].repeat(ROBNERF_PATCH_SIZE_2)

        elif self.config.model.batch_image_sampling:
            index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
        else:
            index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)

        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            if "robust_loss" in self.config.system.loss and self.config.system.loss.robust_loss:
                # while True:
                #     x_ = torch.randint(0 + ROBNERF_PATCH_SIZE_HALF,
                #                        (self.dataset.w - ROBNERF_PATCH_SIZE_HALF) + 1,
                #                        size=(ROBNERF_PATCH_N,),
                #                        device=self.dataset.all_images.device)
                #     y_ = torch.randint(0 + ROBNERF_PATCH_SIZE_HALF,
                #                        (self.dataset.h - ROBNERF_PATCH_SIZE_HALF) + 1,
                #                        size=(ROBNERF_PATCH_N,),
                #                        device=self.dataset.all_images.device)
                #
                #     overlap = False
                #     for blaiter in range(ROBNERF_PATCH_N):
                #         for blaiter2 in range(ROBNERF_PATCH_N):
                #             if blaiter == blaiter2:
                #                 continue
                #             if (index_[blaiter] == index_[blaiter2])\
                #                 and (abs(x_[blaiter] - x_[blaiter2]) < ROBNERF_PATCH_SIZE)\
                #                 and (abs(y_[blaiter] - y_[blaiter2]) < ROBNERF_PATCH_SIZE):
                #                 overlap = True
                #                 print("overlap")
                #                 break
                #         if overlap:
                #             break
                #
                #     if not overlap:
                #         break
                valid = False
                while not valid:
                    x_ = torch.randint(0 + ROBNERF_PATCH_SIZE_HALF,
                                       (self.dataset.w - ROBNERF_PATCH_SIZE_HALF) + 1,
                                       size=(ROBNERF_PATCH_N,),
                                       device=self.dataset.all_images.device)
                    y_ = torch.randint(0 + ROBNERF_PATCH_SIZE_HALF,
                                       (self.dataset.h - ROBNERF_PATCH_SIZE_HALF) + 1,
                                       size=(ROBNERF_PATCH_N,),
                                       device=self.dataset.all_images.device)

                    x = torch.empty(ROBNERF_PATCH_N * ROBNERF_PATCH_SIZE_2, dtype=x_.dtype, device=x_.device)
                    y = torch.empty(ROBNERF_PATCH_N * ROBNERF_PATCH_SIZE_2, dtype=y_.dtype, device=y_.device)
                    for ind in range(ROBNERF_PATCH_N):
                        x[ind * ROBNERF_PATCH_SIZE_2: (ind + 1) * ROBNERF_PATCH_SIZE_2] = torch.arange(
                            x_[ind] - ROBNERF_PATCH_SIZE_HALF,
                            x_[ind] + ROBNERF_PATCH_SIZE_HALF,  # last is exclusive anyhow
                        ).repeat(ROBNERF_PATCH_SIZE)

                        values = torch.arange(
                            y_[ind] - ROBNERF_PATCH_SIZE_HALF,
                            y_[ind] + ROBNERF_PATCH_SIZE_HALF  # last is exclusive anyhow
                        )
                        assert len(values) == 16
                        for ind2 in range(ROBNERF_PATCH_SIZE):
                            offset_outer = ind * ROBNERF_PATCH_SIZE_2
                            offset_inner = ind2 * ROBNERF_PATCH_SIZE
                            offset_inner_pp = (ind2 + 1) * ROBNERF_PATCH_SIZE
                            my_values = values[ind2].repeat(ROBNERF_PATCH_SIZE)
                            y[offset_outer + offset_inner: offset_outer + offset_inner_pp] = my_values

                    # This is sadly neccessary as only for valid rays (opacity > 0) a loss is computed
                    # just skipping the optimization if no ray is valid also does not really help (pt lightning does not really allow)
                    # so it is actually required to make sure at least one foreground ray makes it into the batch
                    # only every about 200 iterations this does not happen naturally through the random sampling
                    # so this hardly has an effect at all
                    # if anything this makes the optimization easier for robust nerf, so we definitely stay scientific
                    # if not self.dataset.all_fg_masks[index, y, x].sum() == 0:
                    #     valid = True
                    valid=True

            else:
                x = torch.randint(
                    0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )
                y = torch.randint(
                    0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])        
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if "robust_loss" in self.config.system.loss and self.config.system.loss.robust_loss:
            # Sometime no ray ends up being valid
            # due to the sparse sampling of the patches compared to the dense uniform sampling with rays
            # and also the resulting more biased updates of the occupancy grid
            self.log('train/occupancy', torch.count_nonzero(out['rays_valid'][..., 0]), prog_bar=True)
            if torch.count_nonzero(out['rays_valid'][..., 0]) == 0:
                # print("No valid ray", flush=True)
                return None

            with torch.no_grad():
                # get the L2 errors for all occupied pixels (Euclidean of color differences)
                # set the error to zero for all unoccupied pixels
                # TODO check that probably we want squared, does not change anything but might introduce numerical instability due to the unneccssary root
                pp_valid_error = torch.linalg.vector_norm(
                    out['comp_rgb'][out['rays_valid'][..., 0]] - batch['rgb'][out['rays_valid'][..., 0]],
                    dim=-1,
                    ord=2
                )

                dtype = out['comp_rgb'].dtype
                device = out['comp_rgb'].device
                pp_error = torch.zeros(size=(out['comp_rgb'].shape[0],), dtype=dtype, device=device)

                pp_error[out['rays_valid'][..., 0]] = pp_valid_error

                # take the median as inlier threshold as mentioned in RonustNeRF
                # only use the occupied pixels to compute the threshold (otherwise it would always be too low)
                # use a higher threshold than median, otherwise RobustNeRF just fits the background
                threshold = torch.quantile(pp_valid_error, q=0.8)

                # obtain inlier mask by applying the defined threshold
                # lt OR EQUAL plus eps to make sure at least one occupied is selected in the end, minor change compared to robustnerf
                pp_inlier = pp_error < threshold
                self.log('train/ppinlier', torch.count_nonzero(pp_inlier), prog_bar=True)

                # transform back to original shape
                # recover from flattened(N, H, W)
                # and introduce singleton dimension for channels for next operation
                recovered = torch.unflatten(pp_inlier, dim=0,
                                            sizes=(ROBNERF_PATCH_N, 1, ROBNERF_PATCH_SIZE, ROBNERF_PATCH_SIZE)
                                            ).type(dtype)

                # smooth obtained mask with 3x3 box filter
                kernel = torch.ones((1, 1, 3, 3), device=device, dtype=dtype) / 9.
                sm_inlier = torch.nn.functional.conv2d(recovered, kernel, padding='same')

                # rebinarize with neighbourhood threshold (0.5)
                b_sm_inlier = sm_inlier > 0.5
                b_sm_inlier = b_sm_inlier.flatten()
                self.log('train/sminlier', torch.count_nonzero(b_sm_inlier), prog_bar=True)

                # for every of the patches(16x16) compute the average
                # if above threshold (0.6), label the inner part (8x8) of the patch as inlier
                # TODO WATCH OUT:
                # In the paper they compute the patch loss based on the smoothed mask from the previous step
                # In their published source code, however, they compute it based on the original per pixel mask
                # We stick to what they use in their published source code
                patch_inlier = torch.mean(recovered, dim=[1, 2, 3])
                b_patch_inlier = patch_inlier > 0.6

                # recover shape and apply to inner 8x8 of patch
                inner_offset = int((ROBNERF_PATCH_SIZE - ROBNERF_INNER_PATCH_SIZE) / 2)
                stop_position = ROBNERF_INNER_PATCH_SIZE + inner_offset - 1

                patch_inlier_mask = torch.zeros_like(recovered, dtype=torch.bool)
                patch_inlier_mask[b_patch_inlier, 0, inner_offset:stop_position, inner_offset:stop_position] = True
                patch_inlier_mask = patch_inlier_mask.flatten()
                self.log('train/pchinlier', torch.count_nonzero(patch_inlier_mask), prog_bar=True)

                # sum all masks to obtain final mask
                # make boolean or something and just and ?
                final_mask = torch.logical_or(pp_inlier, b_sm_inlier)
                final_mask = torch.logical_or(final_mask, patch_inlier_mask)
                self.log('train/inlier', torch.count_nonzero(final_mask), prog_bar=True)

                combined_with_occupancy = torch.logical_and(final_mask, out['rays_valid'][..., 0])
                self.log('train/totalil', torch.count_nonzero(combined_with_occupancy), prog_bar=True)

            loss_rgb = F.mse_loss(out['comp_rgb'][combined_with_occupancy],
                                  batch['rgb'][combined_with_occupancy])

            # TODO problem combination of rays valid and robust nerf mask can lead to no ray valid
            # TODO in general the sampling of patches is very much not compatible with the occupancy grid technique

        else:
            if self.config.model.dynamic_ray_sampling:
                train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                                          self.config.model.max_train_num_rays)

            if "l2_loss" in self.config.system.loss and self.config.system.loss.l2_loss:
                loss_rgb = F.mse_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
            else:
                loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])

        self.log('train/loss_rgb', loss_rgb)

        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):  
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)

            psnr_min = torch.min(torch.stack([o['psnr'] for o in out_set.values()]))
            psnr_max = torch.max(torch.stack([o['psnr'] for o in out_set.values()]))
            psnr_std = torch.std(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr/min', psnr_min, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/max', psnr_max, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/std', psnr_std, prog_bar=True, rank_zero_only=True)

            psnr_90 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .90)
            psnr_95 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .95)
            psnr_99 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .99)
            self.log('test/psnr/perc90', psnr_90, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/perc95', psnr_95, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/perc99', psnr_99, prog_bar=True, rank_zero_only=True)

            psnr_1 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .01)
            psnr_5 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .05)
            psnr_10 = torch.quantile(torch.stack([o['psnr'] for o in out_set.values()]), .1)
            self.log('test/psnr/perc1', psnr_1, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/perc5', psnr_5, prog_bar=True, rank_zero_only=True)
            self.log('test/psnr/perc10', psnr_10, prog_bar=True, rank_zero_only=True)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    
