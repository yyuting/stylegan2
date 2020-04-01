# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import projector
import pretrained_networks
from training import dataset
from training import misc

import skimage.io
import os
import numpy
import skimage.morphology

import OpenEXR
from exr_util import *
import speed_vector

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots, start_from_last=False, old_vars=None, advected_img=None, advected_weight=0.0, advect_speed_mask=None, noise_speed_masks={}, advect_noise_weight=0.0, frames_with_ini_zero_noise=False):
    
    use_zero_noise = False
    
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
   
    advect_noise_vars = []
    if old_vars is not None:
        if len(old_vars) > 1:
            advect_noise_vars = old_vars[1:]
            
    proj.start(targets, advect_last_frame_img=advected_img, advect_last_frame_weight=advected_weight, advect_speed_mask=advect_speed_mask, advect_noise_vars=advect_noise_vars, advect_noise_weight=advect_noise_weight, noise_speed_masks=noise_speed_masks)
    if start_from_last:
        # mode 1 is optimize for latent and noise, but initialize with the value optimized for last frame
        if old_vars is not None:
            tflib.set_vars({proj._dlatents_var: old_vars[0]})
            tflib.set_vars({proj._noise_vars[k]: old_vars[k+1] for k in range(len(proj._noise_vars))})
            if frames_with_ini_zero_noise:
                use_zero_noise = True
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step(use_zero_noise=use_zero_noise)
        if proj.get_cur_step() in snapshot_steps:
            latest_img = proj.get_images()
            misc.save_image_grid(latest_img, png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    
    if old_vars is not None:
        # debug only
        test = 0
    
    print('\r%-30s\r' % '', end='', flush=True)
    ans = tflib.run([proj._dlatents_var] + proj._noise_vars)
    numpy.savez(png_prefix + 'raw_vars.npy', *ans)
    
    final_loss = proj.get_lpips()
    
    return ans, latest_img, final_loss

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        ans, latest_img, _ = project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots, temporal_mode):
    
        
    start_from_last = False
    disable_noise_opt = False
    encourage_advected_last_frame = False
    advected_weight = 0
    advect_noise = False
    mask_speed_vector = False
    advected_still_scale = 1
    first_frame_steps = -1
    first_frame_restarts = -1
    encourage_advected_noise = False
    advect_noise_weight = 0
    advect_noise_mode = ''
    frames_with_ini_zero_noise = False
    mask_noise_speed = False
    zero_constrain_mask_boundary = False
    sweep_from_best_spot = False
    sweep_advected_scale = 1
    
    
    if temporal_mode == 0:
        pass
    elif temporal_mode == 1:
        start_from_last = True
    elif temporal_mode == 2:
        disable_noise_opt = True
    elif temporal_mode == 3:
        start_from_last = True
        disable_noise_opt = True
    elif temporal_mode == 4:
        start_from_last = True
        encourage_advected_last_frame = True
        # TODO: allow weight to be a parameter
        advected_weight = 2e-4
    elif temporal_mode == 5:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        disable_noise_opt = True
    elif temporal_mode == 6:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
    elif temporal_mode == 7:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        disable_noise_opt = True
    elif temporal_mode == 8:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
    elif temporal_mode == 9:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
        disable_noise_opt = True
    elif temporal_mode == 10:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
        advect_noise = True
    elif temporal_mode == 11:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 1
    elif temporal_mode == 12:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
        advect_noise = True
        advect_noise_mode = 'nn'
    elif temporal_mode == 13:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_steps = 2000
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 200
        advect_noise_mode = 'nn'
    elif temporal_mode == 14:
        # weight determined by looking at loss values AFTER the 1000 optimization iterations
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-2
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 10
    elif temporal_mode == 15:
        # weight determined by looking at loss values AFTER the 1000 optimization iterations
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-2
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 10
        frames_with_ini_zero_noise = True
    elif temporal_mode == 16:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 10
        frames_with_ini_zero_noise = True
    elif temporal_mode == 17:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-2
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 1
        frames_with_ini_zero_noise = True
    elif temporal_mode == 18:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-2
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-2
        frames_with_ini_zero_noise = True
    elif temporal_mode == 19:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-4
        frames_with_ini_zero_noise = True
    elif temporal_mode == 20:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 10
    elif temporal_mode == 21:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 2
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-4
    elif temporal_mode == 22:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 4
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-4
        mask_noise_speed = True
    elif temporal_mode == 23:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-4
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 4
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-4
        mask_noise_speed = True
        zero_constrain_mask_boundary = True
    elif temporal_mode == 24:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-2
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 4
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-2
        mask_noise_speed = True
        zero_constrain_mask_boundary = True
    elif temporal_mode == 25:
        start_from_last = True
        encourage_advected_last_frame = True
        advected_weight = 2e-3
        mask_speed_vector = True
        advected_still_scale = 0.5
        first_frame_restarts = 4
        advect_noise = True
        encourage_advected_noise = True
        advect_noise_weight = 2e-3
        mask_noise_speed = True
        zero_constrain_mask_boundary = True
    else:
        raise
        
    exr_pl = None
    xv = None
    yv = None
    
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs, disable_noise_opt=disable_noise_opt, advect_last_frame=encourage_advected_last_frame, mask_speed_vector=mask_speed_vector, encourage_advected_noise=encourage_advected_noise, mask_noise_speed=mask_noise_speed)

    print('Loading images from "%s"...' % dataset_name)
    #dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    #assert dataset_obj.shape == Gs.output_shape[1:]
    
    img_files = sorted([os.path.join(data_dir, dataset_name, file) for file in os.listdir(os.path.join(data_dir, dataset_name)) if file.endswith('.jpg')])
    num_images = len(img_files)
    
    all_vars = []
    noise_confidence = {}
    noise_speed_masks = {}
    old_ans = None
    
    latest_img = None
    advected_img = None
    advect_speed_mask = None
    
    if sweep_from_best_spot:
        current_advected_weight = advected_weight * sweep_advected_scale
        current_advected_noise_weight = advect_noise_weight * sweep_advected_scale
        traversals = [(advected_weight * sweep_advected_scale, advect_noise_weight * sweep_advected_scale, ''),
                      (advected_weight, advect_noise_weight, 'sweep_2_')]
        
    else:
        current_advected_weight = advected_weight
        current_advected_noise_weight = advect_noise_weight
        traversals = [(advected_weight, advect_noise_weight, '')]
                
    best_lpips_across_frames = 1e8
    
    def sweep_fwd_bcw(start_idx, end_idx, opt_args, forward=True, prefix='', latest_img=None):
                
        if forward:
            assert start_idx <= end_idx
            num_images = end_idx - start_idx + 1
            traveral_order = np.arange(start_idx, end_idx + 1)
        else:
            assert start_idx >= end_idx
            num_images = start_idx - end_idx + 1
            traversal_order = np.arange(start_idx, end_idx - 1, -1)
                        
        for image_idx in traversal_order:
            print('Projecting image %s %d/%d ...' % (prefix, image_idx, num_images))
            images = skimage.img_as_float(skimage.io.imread(img_files[image_idx]))
            images = np.expand_dims(np.transpose(images, (2, 0, 1)), 0)
            images = misc.adjust_dynamic_range(images, [0, 1], [-1, 1])
            
            if opt_args.encourage_advected_last_frame and latest_img is not None:
                exrfile = OpenEXR.InputFile(os.path.join(data_dir, dataset_name, 'raw', '%04d.exr' % (image_idx + 1)))
                if forward:
                    speed_vectors = channels_to_ndarray(exrfile, ['vector_x.V', 'vector_y.V'])
                else:
                    speed_vectors = -channels_to_ndarray(exrfile, ['vector_z.V', 'vector_a.V'])
                    
                if speed_vectors.shape[0] != latest_img.shape[2] or speed_vectors.shape[1] != latest_img.shape[3]:
                    if exr_pl is None:
                        exr_pl = np.zeros((latest_img.shape[2], latest_img.shape[3], 2))
                        
                    if xv is None or yv is None:
                        xv, yv = np.meshgrid(np.arange(latest_img.shape[2]), np.arange(latest_img.shape[3]), indexing='ij')
                    
                    h_diff = latest_img.shape[2] - speed_vectors.shape[0]
                    w_diff = latest_img.shape[3] - speed_vectors.shape[1]
                    exr_pl[h_diff // 2: h_diff // 2 + speed_vectors.shape[0], w_diff // 2: w_diff // 2 + speed_vectors.shape[1], :] = speed_vectors[:, :, :]
                    speed_vectors = exr_pl
                    
                advected_img, _ = speed_vector.advect_img(latest_img, -speed_vectors[:, :, 0], -speed_vectors[:, :, 1], xv, yv, is_tensor=True)
                
                if opt_args.zero_constrain_mask_boundary:
                    nonzero_speed = (speed_vectors[:, :, 0] != 0) * (speed_vectors[:, :, 1] != 0)
                    dilation_r = np.ceil(np.max(np.abs(speed_vectors))) + 1
                    selem = skimage.morphology.disk(dilation_r)
                    dilated_speed = skimage.morphology.dilation(nonzero_speed, selem)
                    zero_constrain_mask = dilated_speed
                    zero_constrain_mask[nonzero_speed] = 0
                    
                if opt_args.mask_speed_vector:
                    advect_speed_mask = np.ones((speed_vectors.shape[0], speed_vectors.shape[1]))
                    advect_speed_zero_idx = (speed_vectors[:, :, 0] == 0) * (speed_vectors[:, :, 1] == 0)
                    advect_speed_mask[advect_speed_zero_idx] = opt_args.advected_still_scale

                    if zero_constrain_mask_boundary:
                        advect_speed_mask[zero_constrain_mask] = 0

                    advect_speed_mask = np.expand_dims(np.expand_dims(advect_speed_mask, 0), 0)
                    
    
    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        #images, _labels = dataset_obj.get_minibatch_np(1)
        images = skimage.img_as_float(skimage.io.imread(img_files[image_idx]))
        images = np.expand_dims(np.transpose(images, (2, 0, 1)), 0)
        images = misc.adjust_dynamic_range(images, [0, 1], [-1, 1])
        #images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        
        if encourage_advected_last_frame and latest_img is not None:
            exrfile = OpenEXR.InputFile(os.path.join(data_dir, dataset_name, 'raw', '%04d.exr' % (image_idx + 1)))
            speed_vectors = channels_to_ndarray(exrfile, ['vector_x.V', 'vector_y.V'])
            
            if speed_vectors.shape[0] != latest_img.shape[2] or speed_vectors.shape[1] != latest_img.shape[3]:
                
                if exr_pl is None:
                    exr_pl = np.zeros((latest_img.shape[2], latest_img.shape[3], 2))
                
                if xv is None or yv is None:
                    xv, yv = np.meshgrid(np.arange(latest_img.shape[2]), np.arange(latest_img.shape[3]), indexing='ij')
                
                h_diff = latest_img.shape[2] - speed_vectors.shape[0]
                w_diff = latest_img.shape[3] - speed_vectors.shape[1]
                exr_pl[h_diff // 2: h_diff // 2 + speed_vectors.shape[0], w_diff // 2: w_diff // 2 + speed_vectors.shape[1], :] = speed_vectors[:, :, :]
                speed_vectors = exr_pl
            
            advected_img, _ = speed_vector.advect_img(latest_img, -speed_vectors[:, :, 0], -speed_vectors[:, :, 1], xv, yv, is_tensor=True)
            
            if zero_constrain_mask_boundary:
                nonzero_speed = (speed_vectors[:, :, 0] != 0) * (speed_vectors[:, :, 1] != 0)
                dilation_r = np.ceil(np.max(np.abs(speed_vectors))) + 1
                selem = skimage.morphology.disk(dilation_r)
                dilated_speed = skimage.morphology.dilation(nonzero_speed, selem)
                zero_constrain_mask = dilated_speed
                zero_constrain_mask[nonzero_speed] = 0
            
            if mask_speed_vector:
                
                advect_speed_mask = np.ones((speed_vectors.shape[0], speed_vectors.shape[1]))
                advect_speed_zero_idx = (speed_vectors[:, :, 0] == 0) * (speed_vectors[:, :, 1] == 0)
                advect_speed_mask[advect_speed_zero_idx] = advected_still_scale
                
                if zero_constrain_mask_boundary:
                    advect_speed_mask[zero_constrain_mask] = 0
                
                advect_speed_mask = np.expand_dims(np.expand_dims(advect_speed_mask, 0), 0)
                
            
            if advect_noise:
                
                if zero_constrain_mask_boundary:
                    zero_constrain_mask = np.expand_dims(np.expand_dims(zero_constrain_mask, 0), 0)
                
                noise_speed_masks = {}
                
                downsampled_speed_lookup = {}
                transposed_speed = np.expand_dims(speed_vectors.transpose(2, 0, 1), 0)
                nonzero_speed = ((transposed_speed[:, 0:1, :, :] != 0) * (transposed_speed[:, 1:2, :, :] != 0)).astype('f')
                for noise_idx in range(len(old_ans) - 1):
                    current_noise = old_ans[noise_idx + 1]
                    factor = transposed_speed.shape[2] // current_noise.shape[2]
                    
                    if factor in downsampled_speed_lookup.keys():
                        downsampled_speed, current_xv, current_yv, downsampled_nonzero_speed, downsampled_zero_constrain = downsampled_speed_lookup[factor]
                    else:
                        downsampled_speed = proj.downsample_raw_img(transposed_speed, factor=factor, rescale=False) / factor
                        downsampled_speed = downsampled_speed[0].transpose(1, 2, 0)
                        
                        downsampled_nonzero_speed = proj.downsample_raw_img(nonzero_speed, factor=factor, rescale=False)
                        if zero_constrain_mask_boundary:
                            downsampled_zero_constrain = proj.downsample_raw_img(zero_constrain_mask.astype('f'), factor=factor, rescale=False)
                        else:
                            downsampled_zero_constrain = None
                        current_xv, current_yv = np.meshgrid(np.arange(current_noise.shape[2]), np.arange(current_noise.shape[3]), indexing='ij')
                        downsampled_speed_lookup[factor] = (downsampled_speed, current_xv, current_yv, downsampled_nonzero_speed, downsampled_zero_constrain)
                    
                    advected_current_noise, current_pixel_confidence = speed_vector.advect_img(current_noise, -downsampled_speed[:, :, 0], -downsampled_speed[:, :, 1], current_xv, current_yv, is_tensor=True, advect_mode=advect_noise_mode)
                    
                    if not np.allclose(old_ans[noise_idx + 1], advected_current_noise):
                        # debug only
                        testa = 0
                    
                    old_ans[noise_idx + 1] = advected_current_noise
                    
                    if factor not in noise_confidence.keys():
                        noise_confidence[factor] = current_pixel_confidence
                
                
                    if mask_noise_speed:
                        if int(proj._noise_vars[noise_idx].shape[-1]) in noise_speed_masks.keys():
                            continue
                        current_mask = np.expand_dims(np.expand_dims(current_pixel_confidence, 0), 0) * downsampled_nonzero_speed
                        assert np.max(current_mask) <= 1
                        assert np.min(current_mask) >= 0
                        current_mask = current_mask * (1 - advected_still_scale) + advected_still_scale
                        
                        if zero_constrain_mask_boundary:
                            current_mask *= (1 - downsampled_zero_constrain)
                        
                        noise_speed_masks[int(proj._noise_vars[noise_idx].shape[-1])] = current_mask
                        
        
        if image_idx == 0:
            default_steps = proj.num_steps
            
            if first_frame_steps > 0:
                proj.num_steps = first_frame_steps
            
            if first_frame_restarts > 0:
                restarts = first_frame_restarts
            else:
                restarts = 1
        else:
            proj.num_steps = default_steps
            restarts = 1
        
        min_lpips = 1e8
        ans = None
        
        for _ in range(restarts):
            current_ans, latest_img, loss_val = project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots, start_from_last=start_from_last, old_vars=old_ans, advected_img=advected_img, advected_weight=current_advected_weight, advect_speed_mask=advect_speed_mask, noise_speed_masks=noise_speed_masks, advect_noise_weight=current_advected_noise_weight, frames_with_ini_zero_noise=frames_with_ini_zero_noise)
            current_lpips = loss_val
            if current_lpips < min_lpips:
                min_lpips = current_lpips
                ans = current_ans
                
        if min_lpips < best_lpips_across_frames:
            best_lpips_across_frames = min_lpips
            best_idx = image_idx
        
        old_ans = ans
        all_vars.append(ans)
        
    if False:
        noise_vars_sum = []
        for i in range(len(all_vars)):
            for j in range(1, len(all_vars[i])):
                if i == 0:
                    noise_vars_sum.append(all_noise_vars[i][j])
                else:
                    noise_vars_sum[j-1] += all_noise_vars[i][j]

        for j in range(len(noise_vars_sum)):
            noise_vars_sum[j] /= len(all_noise_vars)

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_images_parser.add_argument('--temporal_mode', type=int, help='Mode to handle temporal coherence, detaulf does nothing', default=0)

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector.project_generated_images',
        'project-real-images': 'run_projector.project_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
