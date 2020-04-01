import numpy
import numpy as np
import sys
import os

from read_npz import read_npz
import pretrained_networks
import projector
import dnnlib.tflib as tflib
from training import misc

import numpy.random

def main():
    
    _G, _D, Gs = pretrained_networks.load_networks('gdrive:networks/stylegan2-car-config-f.pkl')
    proj = projector.Projector()
    proj.set_network(Gs)
    
    dir = sys.argv[1]
    
    avg_noise = False
    linear_interp = False
    endpoint_lin_interp = False
    prefix = ''
    if 'avg_noise' in sys.argv:
        avg_noise = True
        prefix += '_avg_noise'
    if 'linear_interp' in sys.argv:
        linear_interp = True
        prefix += '_linear_interp'
    if 'endpoint_lin_interp' in sys.argv:
        endpoint_lin_interp = True
        prefix += '_endpoint_lin_interp'
        
    if prefix == '':
        prefix = 'sanity'
    elif prefix.startswith('_'):
        prefix = prefix[1:]
    
    raw_var_files = sorted([file for file in os.listdir(dir) if file.endswith('.npz')])
    
    sum_noise_vars = []
    latent_vars = []
    
    for i in range(len(raw_var_files)):
        file = raw_var_files[i]
        raw_var_val = read_npz(os.path.join(dir, file))
        latent_vars.append(raw_var_val[0])
        
        if avg_noise:
            for j in range(1, len(raw_var_val)):
                if i == 0:
                    sum_noise_vars.append(raw_var_val[j])
                else:
                    sum_noise_vars[j-1] += raw_var_val[j]
        else:
            sum_noise_vars.append(raw_var_val[1:])
                
        if False:
            # sanity check: only using raw_var_val should recover projected image
            tflib.set_vars({proj._dlatents_var: raw_var_val[0]})
            tflib.set_vars({proj._noise_vars[k]: raw_var_val[k+1] for k in range(len(proj._noise_vars))})
            misc.save_image_grid(proj.get_images(), os.path.join(dir, 'sanity_%05d.png' % i), drange=[-1,1])
            
        if False:
            # another sanity check: randomly sample noise to see what's happening
            tflib.set_vars({proj._dlatents_var: raw_var_val[0]})
            rnd = np.random.RandomState(0)
            
            for j in range(30):
                tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in proj._noise_vars})
                misc.save_image_grid(proj.get_images(), os.path.join(dir, 'sanity_random_noise_%05d_%05d.png' % (i, j)), drange=[-1,1])
            return
            
    if avg_noise:
        for j in range(len(sum_noise_vars)):
            sum_noise_vars[j] /= len(raw_var_files)

        tflib.set_vars({proj._noise_vars[k]: sum_noise_vars[k] for k in range(len(proj._noise_vars))})
      
    n = len(latent_vars)
    
    if linear_interp:
        A = n * (2 * n - 1) / (6 * (n - 1))
        B = n
        C = 0
        D = 0
        E = n
        for i in range(n):
            C += i * latent_vars[i]
            D += latent_vars[i]
        C *= -2 / (n - 1)
        D *= -2
        
        factor = 4 * A * B - E * E
        
        k_min = (2 * B * C - D * E) / factor
        b_min = (2 * A * D - C * E) / factor
    
    tflib.run(proj._noise_normalize_op)
    
    for i in range(len(latent_vars)):
        if linear_interp:
            current_latent = i * k_min / (n - 1) + b_min
            tflib.set_vars({proj._dlatents_var: current_latent})
        elif endpoint_lin_interp:
            current_latent = i / (n - 1) * (latent_vars[n-1] - latent_vars[0]) + latent_vars[0]
            tflib.set_vars({proj._dlatents_var: current_latent})
        else:
            tflib.set_vars({proj._dlatents_var: latent_vars[i]})
        if not avg_noise:
            tflib.set_vars({proj._noise_vars[k]: sum_noise_vars[i][k] for k in range(len(proj._noise_vars))})
        misc.save_image_grid(proj.get_images(), os.path.join(dir, '%s_%05d.png' % (prefix, i)), drange=[-1,1])
        
        
if __name__ == '__main__':
    main()
        
    