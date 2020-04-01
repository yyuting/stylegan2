import tensorflow as tf
import sys
import os
import numpy
import numpy as np
import skimage
import skimage.io
import dnnlib.tflib as tflib

def generate_metric(metric, drange=1, nbatches=1):
    img0 = tf.placeholder(tf.float32, (nbatches, None, None, 3))
    img1 = tf.placeholder(tf.float32, (nbatches, None, None, 3))
    imgs = [img0, img1]
    if metric == 'ssim':
        metric = tf.image.ssim(img0, img1, drange)
    elif metric == 'l2':
        metric = tf.reduce_mean((img0 - img1) ** 2, (1, 2, 3))
    elif metric == 'symmetric_l2':
        img2 = tf.placeholder(tf.float32, (nbatches, None, None, 3))
        imgs.append(img2)
        metric = tf.reduce_mean((tf.sign((img0 - img1) * (img2 - img1)) + 1.5) * ((img0 - img1) ** 2 + (img1 - img2) ** 2), (1, 2, 3))
    elif metric == 'symmetric':
        img2 = tf.placeholder(tf.float32, (nbatches, None, None, 3))
        imgs.append(img2)
        metric = tf.reduce_mean(tf.clip_by_value(tf.sign((img0 - img1) * (img2 - img1)), 0, 1), (1, 2, 3))
    else:
        raise
    return metric, imgs

def main():
    basedir = sys.argv[1]
    filename_pattern = sys.argv[2]
    start_f = int(sys.argv[3])
    end_f = int(sys.argv[4])
    metrics = sys.argv[5].split(',')
    
    img_files = []
    for ind in range(start_f, end_f + 1):
        img_files.append(os.path.join(basedir, filename_pattern % ind))
    
    for metric in metrics:
        
        metric_tensor, imgs = generate_metric(metric)
        tflib.init_tf()
        
        feed_dict = {}
        for img in imgs:
            feed_dict[img] = None
            
        first = None
        last = None
        all_vals = np.empty(len(img_files) - 1)

        for ind in range(len(img_files) - len(imgs) + 1):
            
            for img_ind in range(len(imgs) - 1):
                if feed_dict[imgs[img_ind]] is None:
                    feed_dict[imgs[img_ind]] = np.expand_dims(skimage.img_as_float(skimage.io.imread(img_files[ind + img_ind])), 0)
                else:
                    feed_dict[imgs[img_ind]] = feed_dict[imgs[img_ind+1]]
            feed_dict[imgs[-1]] = np.expand_dims(skimage.img_as_float(skimage.io.imread(img_files[ind + len(imgs) - 1])), 0)
            
            if False:
                if first is None:
                    first = skimage.img_as_float(skimage.io.imread(img_files[ind]))
                else:
                    first = last
                last = skimage.img_as_float(skimage.io.imread(img_files[ind+1]))
                val = tflib.run(metric_tensor, feed_dict={img0: np.expand_dims(first, 0), img1: np.expand_dims(last, 0)})
            
            val = tflib.run(metric_tensor, feed_dict=feed_dict)
            all_vals[ind] = val[0]
            
            if len(imgs) == 2:
                diff = ((feed_dict[imgs[0]] - feed_dict[imgs[1]]) + 1) / 2
            elif len(imgs) == 3:
                #diff = (np.sign((feed_dict[imgs[0]] - feed_dict[imgs[1]]) * (feed_dict[imgs[2]] - feed_dict[imgs[1]])) + 1) / 2
                diff = np.clip((np.sign((feed_dict[imgs[0]] - feed_dict[imgs[1]]) * (feed_dict[imgs[2]] - feed_dict[imgs[1]]))), 0, 1)
            #diff = ((first - last) + 1) / 2
            skimage.io.imsave(os.path.join(basedir, 'adjacent_diff_%dD_%05d.png' % (len(imgs), ind)), diff[0])
            
            print(ind)

        numpy.save(os.path.join(basedir, metric + '.npy'), all_vals)
        print(np.mean(all_vals), np.var(all_vals))
        open(os.path.join(basedir, metric + '.txt'), 'w').write('avg across all frames: %f\nvar across all frames: %f' % (np.mean(all_vals), np.var(all_vals)))
                


if __name__ == '__main__':
    main()