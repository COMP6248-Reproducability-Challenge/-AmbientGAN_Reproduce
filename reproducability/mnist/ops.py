from glob import glob
import os
import struct
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from config import *

from measurement import *
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


class RealDsIterator(object):

    def __init__(self):
        self.mnist = input_data.read_data_sets('./data/mnist', one_hot=True, validation_size=0)

    def next_batch(self, batch_size=64):
        x_real, y_real = self.mnist.train.next_batch(batch_size)
        x_real = x_real.reshape(batch_size, 28, 28, 1)

        return [x_real, y_real]



# function to get training data
def load_train_data(args):
    data = input_data.read_data_sets('./data/mnist', one_hot=True, validation_size=0)
    train_batch, label_batch = data.train.next_batch(args.batch_size)
    label_batch=tf.cast(label_batch, dtype=tf.float32)
    train_batch = train_batch.reshape(args.batch_size, 28, 28, 1)

    # paths = os.path.join(args.data, "./data/mnist/*.jpg")
    # label_file = 'train-labels.idx1-ubyte'
    # label_file_size = 60008
    # label_file_size = str(label_file_size - 8) + 'B'
    # label_buf = open(label_file, 'rb').read()
    # labels = struct.unpack_from(
    #     '>' + label_file_size, label_buf, struct.calcsize('>II'))
    # labels = np.array(labels).astype(np.int64)
    # depth = 10
    # y = tf.one_hot(labels, depth)
    data_count = 60000
    #
    # filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths),shuffle=False)
    #
    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(filename_queue)
    # images = tf.image.decode_jpeg(image_file, channels=1)
    #
    # # input image range from -1 to 1
    # # center crop 32x32 since raw images are not center cropped.
    # # images = tf.image.central_crop(images, 0.5)
    # images = tf.image.resize_images(images, [args.input_height, args.input_width])
    #
    # images = tf.image.convert_image_dtype(images, dtype=tf.float32)
             # / 127.5 - 1
    images=tf.convert_to_tensor(train_batch)

    # apply measurement models
    if args.measurement == "block_pixels":
        images = block_pixels(images, probability=args.prob)
    elif args.measurement == "block_patch":
        images = block_patch(images, patch_size=args.patch_size)
    elif args.measurement == "keep_patch":
        images = keep_patch(images, patch_size=args.patch_size)
    elif args.measurement == "conv_noise":
        images = conv_noise(images, kernel_size=args.kernel_size, stddev=args.stddev)
    # added by rick
    elif args.measurement == "block_pixels_patch":
        images = block_pixels_patch(images, probability=args.prob, patch_size=args.patch_size)
    train_batch=images


    # train_batch= tf.train.shuffle_batch([images],
    #                                  batch_size=args.batch_size,
    #                                  capacity=args.batch_size * 2,
    #                                  min_after_dequeue=args.batch_size
    #                                  )

    return train_batch, data_count, label_batch


# function to save images in tile
# comment this function block if you don't have opencv
def img_tile(epoch, step, args, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None
    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    # cv2.imwrite(args.images_path + "/img_" + str(epoch) +str(step)+ ".jpg", (tile_img + 1)*127.5)
    plt.imsave(args.images_path + "/img_" + str(epoch) + str(step) + ".png", np.ndarray.squeeze(tile_img, 2),
               cmap='gray')


# function to save images in tile
# comment this function block if you don't have opencv
def img_tile_2(epoch, step, args, imgs, aspect_ratio=1.0, tile_shape=None, border=1, border_color=0):
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None
    # Grid shape
    img_shape = np.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(np.ceil(np.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(np.ceil(np.sqrt(n_imgs / aspect_ratio)))
        grid_shape = np.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = np.array(tile_shape)

    # Tile image shape
    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break
            img = imgs[img_idx]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img

    # cv2.imwrite(args.images_path + "/train_img_" + str(epoch) +str(step)+ ".jpg", (tile_img + 1)*127.5)
    plt.imsave(args.images_path + "/train_img_" + str(epoch) +str(step)+ ".png", \
               np.ndarray.squeeze(tile_img, 2), cmap='gray')


