from exr_util import *
import numpy
import numpy as np
import os
import sys
import skimage
import skimage.io
import skimage

def get_speed_vectors(file):
    exrfile = OpenEXR.InputFile(file)
    speed_vectors = channels_to_ndarray(exrfile, ['vector_x.V', 'vector_y.V', 'vector_z.V', 'vector_a.V'])
    return speed_vectors

def pixel_in_range(xv, yv, h_max, w_max):
    return ((xv >= 0) * (xv <= h_max) * (yv >= 0) * (yv <= w_max)).astype('f')

def advect_img(img, speed_x, speed_y, xv, yv, ref_img=None, is_tensor=False, fill=None, advect_mode=''):

    # using bilinear sampling
    new_xv = xv + speed_y
    new_yv = yv - speed_x
    xv_low = np.floor(new_xv).astype('i')
    xv_hi = np.ceil(new_xv).astype('i')
    yv_low = np.floor(new_yv).astype('i')
    yv_hi = np.ceil(new_yv).astype('i')

    ratio_x_low = xv_low + 1 - new_xv
    ratio_y_low = yv_low + 1 - new_yv
    
    if is_tensor:
        h_max = img.shape[2] - 1
        w_max = img.shape[3] - 1
        
    else:
        h_max = img.shape[0] - 1
        w_max = img.shape[1] - 1
        
    pixel_confidence = pixel_in_range(xv_low, yv_low, h_max, w_max) * ratio_x_low * ratio_y_low + \
                       pixel_in_range(xv_low, yv_hi, h_max, w_max) * ratio_x_low * (1 - ratio_y_low) + \
                       pixel_in_range(xv_hi, yv_low, h_max, w_max) * (1 - ratio_x_low) * ratio_y_low + \
                       pixel_in_range(xv_hi, yv_hi, h_max, w_max) * (1 - ratio_x_low) * (1 - ratio_y_low)
        
    if np.sum(pixel_confidence) < pixel_confidence.size:
        # do something for out of range indices
        # identify the max and min indices, then expand img with corresponding shape (we can keep [0, 0] the same and onoly expand outwards, because the negative indices can automacially wrap around)
        extra_h = max(np.max(xv_hi) - h_max, 0) + max(-np.min(xv_low), 0)
        extra_w = max(np.max(yv_hi) - w_max, 0) + max(-np.min(yv_low), 0)
        
        if is_tensor:
            expanded_shape = (img.shape[0], img.shape[1], img.shape[2] + extra_h, img.shape[3] + extra_w)
        else:
            expanded_shape = (img.shape[0] + extra_h, img.shape[1] + extra_w, img.shape[2])
        
        if fill is not None:
            expanded_img = fill * np.ones(expanded_shape)
        else:
            expanded_img = np.random.randn(*expanded_shape)
        
        if is_tensor:
            expanded_img[:, :, :h_max+1, :w_max+1] = img[:, :, :, :]
        else:
            expanded_img[:h_max+1, :w_max+1, :] = img[:, :, :]
        
    else:
        expanded_img = img
        

    if advect_mode == 'nn':
        x_hit_low = (ratio_x_low > 0.5).astype('f')
        y_hit_low = (ratio_y_low > 0.5).astype('f')
        xv_nn = (x_hit_low * xv_low + (1 - x_hit_low) * xv_hi).astype('i')
        yv_nn = (y_hit_low * yv_low + (1 - y_hit_low) * yv_hi).astype('i')
        
    if is_tensor:
        if advect_mode == '':
            new_img = expanded_img[:, :, xv_low, yv_low] * ratio_x_low * ratio_y_low + \
                      expanded_img[:, :, xv_low, yv_hi] * ratio_x_low * (1 - ratio_y_low) + \
                      expanded_img[:, :, xv_hi, yv_low] * (1 - ratio_x_low) * ratio_y_low + \
                      expanded_img[:, :, xv_hi, yv_hi] * (1 - ratio_x_low) * (1 - ratio_y_low)
        elif advect_mode == 'nn':
            new_img = expanded_img[:, :, xv_nn, yv_nn]
    else:
        if advect_mode == '':
            new_img = expanded_img[xv_low, yv_low, :] * np.expand_dims(ratio_x_low * ratio_y_low, 2) + \
                      expanded_img[xv_low, yv_hi, :] * np.expand_dims(ratio_x_low * (1 - ratio_y_low), 2) + \
                      expanded_img[xv_hi, yv_low, :] * np.expand_dims((1 - ratio_x_low) * ratio_y_low, 2) + \
                      expanded_img[xv_hi, yv_hi, :] * np.expand_dims((1 - ratio_x_low) * (1 - ratio_y_low), 2)
        elif advect_mode == 'nn':
            new_img = expanded_img[xv_nn, yv_nn, :]

    if ref_img is not None:
        still_pix = ((speed_y == 0) * (speed_x == 0))
        if is_tensor:
            still_pix = np.tile(np.expand_dims(np.expand_dims(still_pix, 0), 0), [ref_img.shape[0], ref_img.shape[1], 1, 1])
        else:
            still_pix = np.tile(np.expand_dims(still_pix, 2), [1, 1, ref_img.shape[2]])
        new_img[still_pix] = ref_img[still_pix]

    return new_img, pixel_confidence    


def main():
    base_dir = sys.argv[1]
    nfiles = int(sys.argv[2])
    
    if len(sys.argv) > 4:
        img_format = sys.argv[3]
        start_with = int(sys.argv[4])
    else:
        img_format = '%04d.png'
        start_with = 1
        
    if len(sys.argv) > 5:
        exr_dir = sys.argv[5]
    else:
        exr_dir = base_dir
        
    if '--auto_fill_exr' in sys.argv:
        auto_fill_exr = True
    else:
        auto_fill_exr = False

    imgs = [skimage.img_as_float(skimage.io.imread(os.path.join(base_dir, img_format % start_with))),
            skimage.img_as_float(skimage.io.imread(os.path.join(base_dir, img_format % (start_with + 1))))]

    xv, yv = np.meshgrid(np.arange(imgs[0].shape[0]), np.arange(imgs[0].shape[1]), indexing='ij')
    
    if auto_fill_exr:
        exr_pl = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 4))

    for i in range(2, nfiles):
        imgs.append(skimage.img_as_float(skimage.io.imread(os.path.join(base_dir, img_format % (start_with + i)))))

        exrfile = OpenEXR.InputFile(os.path.join(exr_dir, '%04d.exr' % i))
        speed_vectors = channels_to_ndarray(exrfile, ['vector_x.V', 'vector_y.V', 'vector_z.V', 'vector_a.V'])
        
        if auto_fill_exr:
            h_diff = imgs[0].shape[0] - speed_vectors.shape[0]
            w_diff = imgs[0].shape[1] - speed_vectors.shape[1]
            exr_pl[h_diff // 2: h_diff // 2 + speed_vectors.shape[0], w_diff // 2: w_diff // 2 + speed_vectors.shape[1], :] = speed_vectors[:, :, :]
            speed_vectors = exr_pl

        img_before, _ = advect_img(imgs[i-2], -speed_vectors[:, :, 0], -speed_vectors[:, :, 1], xv, yv, imgs[i-1])
        img_after, _ = advect_img(imgs[i], speed_vectors[:, :, 2], speed_vectors[:, :, 3], xv, yv, imgs[i-1])
        skimage.io.imsave(os.path.join(base_dir, '%04d_before.png' % i), img_before)
        skimage.io.imsave(os.path.join(base_dir, '%04d_after.png' % i), img_after)
        
        img_before, _ = advect_img(imgs[i-2], -speed_vectors[:, :, 0], -speed_vectors[:, :, 1], xv, yv)
        img_after, _ = advect_img(imgs[i], speed_vectors[:, :, 2], speed_vectors[:, :, 3], xv, yv)
        skimage.io.imsave(os.path.join(base_dir, '%04d_before_no_ref.png' % i), img_before)
        skimage.io.imsave(os.path.join(base_dir, '%04d_after_no_ref.png' % i), img_after)

if __name__ == '__main__':
    main()
