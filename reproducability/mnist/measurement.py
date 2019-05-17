import tensorflow as tf
import numpy as np
import scipy.stats as st


# Each pixel is independently set to zero with probability p
def block_pixels(input, probability=0.5):
    print('block_pixels...', probability)
    shape = input.get_shape().as_list()

    # for training images
    if len(shape) == 3:
        #change channel
        prob = tf.random_uniform([shape[0], shape[1], 1], minval=0, maxval=1, dtype=tf.float32)
        # prob = tf.tile(prob, [1, 1, 3])
        prob = tf.to_int32(prob < probability)
        prob = tf.cast(prob, dtype=tf.float32)
        res = tf.multiply(input, prob)
    # for generated images
    else:
        res = []
        for idx in range(0, shape[0]):
            #shape error
            prob = tf.random_uniform([shape[1], shape[2], 1], minval=0, maxval=1, dtype=tf.float32)
            # prob = tf.tile(prob, [1, 1, 3])
            prob = tf.to_int32(prob < probability)
            prob = tf.cast(prob, dtype=tf.float32)
            res.append(tf.multiply(input[idx], prob))
        res = tf.stack(res)
    return res


# Gaussian Kernel is used to blur images + noise added
def conv_noise(input, kernel_size=3, stddev=0.0):
    print('conv_noise...', kernel_size, stddev)
    shape = input.get_shape().as_list()

    def gauss_kernel(kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        out_filter = tf.Variable(tf.convert_to_tensor(out_filter), name="filter")
        return out_filter

    convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding="SAME")

    # apply gaussian kernel here
    # for training images
    if len(shape) == 3:
        gauss_kernel = gauss_kernel(kernlen=kernel_size, channels=3)

        input = input[tf.newaxis, ...]
        res = convolve(input, gauss_kernel)
        res = tf.squeeze(res, axis=0)
    # for generated images
    else:
        gauss_kernel = gauss_kernel(kernlen=kernel_size, channels=1) # change channel
        res = convolve(input, gauss_kernel)

    # add noise stddev
    noise = tf.random_normal(shape=tf.shape(res), mean=0.0, stddev=stddev, dtype=tf.float32)

    return res + noise


# A randomly chosen k x k patch is set to zero
def block_patch(input, patch_size=14):
    print('block_patch...', patch_size)
    shape = input.get_shape().as_list()

    # for training images
    if len(shape) == 3:
        patch = tf.zeros([patch_size, patch_size, shape[-1]], dtype=tf.float32)

        rand_num = tf.random_uniform([2], minval=0, maxval=shape[0] - patch_size, dtype=tf.int32)
        h_, w_ = rand_num[0], rand_num[1]

        padding = [[h_, shape[0] - h_ - patch_size], [w_, shape[1] - w_ - patch_size], [0, 0]]
        padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

        res = tf.multiply(input, padded)
    # for generated images
    else:
        patch = tf.zeros([patch_size, patch_size, shape[-1]], dtype=tf.float32)

        res = []
        for idx in range(0, shape[0]):
            rand_num = tf.random_uniform([2], minval=0, maxval=shape[1] - patch_size, dtype=tf.int32) #change shape
            h_, w_ = rand_num[0], rand_num[1]

            padding = [[h_, shape[1] - h_ - patch_size], [w_, shape[2] - w_ - patch_size], [0, 0]] #change shape
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

            res.append(tf.multiply(input[idx], padded))
        res = tf.stack(res)

    return res


# All pixels outside a randomly chosen k x k patch are set to zero
def keep_patch(input, patch_size=14):
# def block_pixels(input, patch_size=32):
    print('keep_patch...', patch_size)
    shape = input.get_shape().as_list()
    # for training images
    if len(shape) == 3:
        # generate a patch
        patch = tf.ones([patch_size, patch_size, shape[-1]], dtype=tf.float32)

        # add padding of 0 randomly to all sides (size should not be greater than the image)
        rand_num = tf.random_uniform([2], minval=0, maxval=shape[0] - patch_size, dtype=tf.int32)
        h_, w_ = rand_num[0], rand_num[1]
        padding = [[h_, shape[0] - h_ - patch_size], [w_, shape[1] - w_ - patch_size], [0, 0]]
        padded = tf.pad(patch, padding, "CONSTANT", constant_values=0)
        res = tf.multiply(input, padded)

    # for generated images
    else:
        patch = tf.ones([patch_size, patch_size, shape[-1]], dtype=tf.float32)

        res = []
        for idx in range(0, shape[0]):
            rand_num = tf.random_uniform([2], minval=0, maxval=shape[1] - patch_size, dtype=tf.int32) #change shape
            h_, w_ = rand_num[0], rand_num[1]

            padding = [[h_, shape[1] - h_ - patch_size], [w_, shape[2] - w_ - patch_size], [0, 0]] # change shape
            padded = tf.pad(patch, padding, "CONSTANT", constant_values=0)

            res.append(tf.multiply(input[idx], padded))
        res = tf.stack(res)

    return res


#Each pixel is independently set to zero with probability p and then patch
def block_pixels_patch(input, probability=0.5, patch_size=32):
    shape = input.get_shape().as_list()
    print('block_pixels_patch...', probability, patch_size)

    #Each pixel is independently set to zero with probability
    res = block_pixels(input, probability)

    res = block_patch(res, patch_size)

    return res