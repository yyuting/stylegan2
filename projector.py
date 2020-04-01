# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import misc

#----------------------------------------------------------------------------

class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = False
        self.clone_net                  = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None
        self._advect_last_frame     = False
        self._disable_noise_opt     = False
        self._mask_speed_vector     = False
        self._encourage_advected_noise = False
        self._mask_noise_speed      = False
        self._masked_noise_speed_weights = {}

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, Gs, minibatch_size=1, disable_noise_opt=False, advect_last_frame=False, mask_speed_vector=False, encourage_advected_noise=False, mask_noise_speed=False):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :1, :] # [N, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Image output graph.
        self._info('Building image output graph...')
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in
        self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, self._Gs.components.synthesis.input_shape[1], 1])
        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        proc_images_expr = (self._images_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

        # Loss graph.
        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            self._lpips = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl')
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        self._lpips_loss = tf.reduce_sum(self._dist)
        self._loss = self._lpips_loss

        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._reg_loss = reg_loss * self.regularize_noise_weight
        self._loss += self._reg_loss
        
        if advect_last_frame:
            self._advect_last_frame = True
            self._advect_last_frame_weight = tf.Variable(0.0, name='advect_last_frame_weight')
            self._advect_last_frame_img = tf.Variable(tf.zeros(proc_images_expr.shape), name='advect_last_frame_img')
            # use L2 loss for now
            self._advected_l2_raw = (proc_images_expr - self._advect_last_frame_img) ** 2
            if mask_speed_vector:
                self._mask_speed_vector = True
                # mask should already be processed to represent weight
                # zero speed should have lower weight
                # nonzero speed should have weight 1
                # creation of weight mask should be handled by start()
                self._speed_mask = tf.Variable(tf.zeros([proc_images_expr.shape[0], 1, proc_images_expr.shape[2], proc_images_expr.shape[3]]), name='advect_speed_mask')
                self._advected_loss = tf.reduce_mean(self._advected_l2_raw * self._speed_mask) * self._advect_last_frame_weight
            else:
                self._advected_loss = tf.reduce_mean(self._advected_l2_raw) * self._advect_last_frame_weight
            self._loss += self._advected_loss
            
        if encourage_advected_noise:
            self._encourage_advected_noise = True
            self._advect_noise_weight = tf.Variable(0.0, name='advect_noise_weight')
            self._advect_noise_vars = []
            self._advect_noise_loss = 0.0
            noise_pix_total = 0
            if mask_noise_speed:
                self._mask_noise_speed = True
                # for highest res, use the same mask as speed, but for lower res, use different masks in various res and incorporates out of bounds confidence
                self._masked_noise_speed_weights = {}
            for noise_idx in range(len(self._noise_vars)):
                var = self._noise_vars[noise_idx]
                self._advect_noise_vars.append(tf.Variable(tf.zeros(var.shape), name='advect_noise_vars%d' % noise_idx))
                raw_advect_noise = (self._advect_noise_vars[-1] - var) ** 2
                if self._mask_noise_speed:
                    if int(var.shape[-1]) in self._masked_noise_speed_weights.keys():
                        mask = self._masked_noise_speed_weights[int(var.shape[-1])]
                    else:
                        mask = tf.Variable(tf.zeros(var.shape), name='advect_noise_mask_%d' % noise_idx)
                        self._masked_noise_speed_weights[int(var.shape[-1])] = mask
                    raw_advect_noise *= mask
                self._advect_noise_loss += tf.reduce_sum(raw_advect_noise)
                noise_pix_total += int(var.shape[2]) * int(var.shape[3])
            self._advect_noise_loss /= noise_pix_total
            self._advect_noise_loss *= self._advect_noise_weight
            self._loss += self._advect_noise_loss
            

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        if disable_noise_opt:
            self._disable_noise_opt = True
            self._opt.register_gradients(self._loss, [self._dlatents_var])
        else:
            self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.images = self.get_images()
        return pres
    
    def downsample_raw_img(self, img, rescale=True, factor=-1):
        """
        downsample img to the size of target_images_var
        """
        img = np.asarray(img, dtype='float32')
        if rescale:
            img = (img + 1) * (255 / 2)
        sh = img.shape
        assert sh[0] == self._minibatch_size
        if factor < 0:
            if sh[2] > self._target_images_var.shape[2]:
                factor = sh[2] // self._target_images_var.shape[2]
                img = np.reshape(img, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
        else:
            img = np.reshape(img, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
        return img
        
    def start(self, target_images, advect_last_frame_weight=0.0, advect_last_frame_img=None, advect_speed_mask=None, advect_noise_vars=[], advect_noise_weight=0.0, noise_speed_masks={}):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = self.downsample_raw_img(target_images)

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        
        tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])})
        
        if self._advect_last_frame:
            if advect_last_frame_img is None:
                advect_last_frame_weight = 0.0
                tflib.set_vars({self._advect_last_frame_img: target_images})
            else:
                advect_last_frame_img = self.downsample_raw_img(advect_last_frame_img)
                tflib.set_vars({self._advect_last_frame_img: advect_last_frame_img})
            
            if self._mask_speed_vector:
                if advect_speed_mask is None:
                    advect_speed_mask = np.ones((target_images.shape[0], 1, target_images.shape[2], target_images.shape[3]))
                else:
                    advect_speed_mask = self.downsample_raw_img(advect_speed_mask, rescale=False)
                tflib.set_vars({self._speed_mask: advect_speed_mask})
            
            tflib.set_vars({self._advect_last_frame_weight: advect_last_frame_weight})
            
        if self._encourage_advected_noise:
            # assert noise are all in correct shape
            
            set_dict = {}
            if len(advect_noise_vars) == 0:
                advect_noise_weight = 0.0
                for noise_idx in range(len(self._advect_noise_vars)):
                    set_dict[self._advect_noise_vars[noise_idx]] = np.zeros(self._advect_noise_vars[noise_idx].shape)
            else:
                assert len(advect_noise_vars) == len(self._noise_vars)
                for noise_idx in range(len(self._advect_noise_vars)):
                    set_dict[self._advect_noise_vars[noise_idx]] = advect_noise_vars[noise_idx]
            
            set_dict[self._advect_noise_weight] = advect_noise_weight
            
            if self._mask_noise_speed:
                if len(noise_speed_masks) == 0:
                    for val in self._masked_noise_speed_weights.values():
                        set_dict[val] = np.zeros(val.shape)
                else:
                    for key in noise_speed_masks.keys():
                        set_dict[self._masked_noise_speed_weights[key]] = noise_speed_masks[key]
            
            tflib.set_vars(set_dict)
        else:
            tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self, use_zero_noise=False):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        if use_zero_noise:
            noise_strength = 0.0
            
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)

        # Print status.
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})
    
    def get_lpips(self):
        return tflib.run(self._lpips_loss, {self._noise_in: 0})

#----------------------------------------------------------------------------
