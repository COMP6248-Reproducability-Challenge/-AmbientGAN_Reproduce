import tensorflow as tf
import numpy as np

# import tflib as lib
from ops import *
from architecture import *
from measurement import *


class ambientGAN():
    def __init__(self, args, mode):
        self.measurement = args.measurement
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.image_dims = [28, 28, 1]
        self.y_dims = 10
        self.data_count = 60000

        # measurement model param setting
        self.prob = args.prob
        self.patch_size = args.patch_size
        self.kernel_size = args.kernel_size
        self.stddev = args.stddev

        # # prepare training data ************
        # self.Y_r, self.data_count, self.label = load_train_data(args)
        # # **********************************
        
        self.train=mode
        self.build_model()
        self.build_loss()

        # summary
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.Y_r_sum = tf.summary.image("input_img", self.Y_r, max_outputs=5)
        self.X_g_sum = tf.summary.image("X_g", self.X_g, max_outputs=5)
        self.Y_g_sum = tf.summary.image("Y_g", self.Y_g, max_outputs=5)

    # structure of the model
    def build_model(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], name="z")
        self.X_o = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='X_o')
        self.Y_r = self.measurement_fn(self.X_o, name="measurement_fn")
        self.label = tf.placeholder(tf.float32, [self.batch_size, self.y_dims], name='label')

        self.X_g, self.g_nets = self.generator(self.z, self.label, self.train, name="generator")

        self.Y_g = self.measurement_fn(self.X_g, name="measurement_fn")
        self.fake_d_logits, self.fake_d_net = self.discriminator(self.Y_g, self.label,self.train,name="discriminator")
        self.real_d_logits, self.real_d_net = self.discriminator(self.Y_r, self.label,self.train,name="discriminator", reuse=True)

        trainable_vars = tf.trainable_variables()
        self.g_vars = []
        self.d_vars = []
        for var in trainable_vars:
            if "generator" in var.name:
                self.g_vars.append(var)
            else:
                self.d_vars.append(var)

    # loss function
    def build_loss(self):
        def calc_loss(logits, label):
            if label == 1:
                y = tf.ones_like(logits)
            else:
                y = tf.zeros_like(logits)
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        self.real_d_loss = calc_loss(self.real_d_logits, 1)
        self.fake_d_loss = calc_loss(self.fake_d_logits, 0)

        self.d_loss = self.real_d_loss + self.fake_d_loss
        self.g_loss = calc_loss(self.fake_d_logits, 1)

    def conv_cond_concat(slef, h, y_label):
        x_shapes = h.get_shape()
        y_shapes = y_label.get_shape()
        return tf.concat([h, y_label * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    # G network from DCGAN
    def generator(self, z, label, train, name="generator"):

        nets = []
        s=28
        gf_dim=64
        gfc_dim=1024
        with tf.variable_scope(name) as scope:
            s2,s4=int(s/2),int(s/4)
            yb = tf.reshape(label, [self.batch_size, 1, 1, 10])
            z=tf.concat([z,label],1)
            g_bn0=batch_norm_linear(name='g_bn0')
            h0 = linear(z, gfc_dim,name="linear")
            h0 = g_bn0(h0,train=train)
            h0=tf.nn.relu(h0)
            nets.append(h0)
            h0=tf.concat([h0,label],1)

            g_bn1=batch_norm_linear(name='g_bn1')
            h1=linear(h0,gf_dim*2*s4*s4, 'g_hi_lin')
            h1=g_bn1(h1,train=train)
            h1=tf.nn.relu(h1)
            nets.append(h1)
            h1=tf.reshape(h1, [self.batch_size,s4,s4,gf_dim*2])
            h1=self.conv_cond_concat(h1, yb)

            g_bn2 = batch_norm_linear(name='g_bn2')
            h2=deconv2d(h1, [self.batch_size,s2,s2,gf_dim*2], name='deconv1')
            h2=g_bn2(h2,train=train)
            h2=tf.nn.relu(h2)
            nets.append(h2)
            h2=self.conv_cond_concat(h2,yb)

            h3=deconv2d(h2,[self.batch_size, s, s, 1], name='deconv2') #change channel
            x_gen=tf.nn.sigmoid(h3)
            nets.append(x_gen)

            return x_gen, nets

    def lrelu(self,x, leak=0.2, name='lrelu'):
        return tf.maximum(x, leak * x, name=name)

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv

    def discriminator(self, input,label, train, name="discriminator", reuse=False):
        nets = []
        df_dim=64
        dfc_dim=1024

        with tf.variable_scope(name, reuse=reuse) as scope:
            yb=tf.reshape(label, [self.batch_size,1,1,10])
            x=self.conv_cond_concat(input, yb)

            h0=self.lrelu(self.conv2d(x, 1+10), name='d_h0_conv')
            nets.append(h0)
            h0=self.conv_cond_concat(h0,yb)

            d_bn1=batch_norm_linear(name='d_bn1')
            h1=self.conv2d(h0, df_dim+10, name='d_h1_conv')
            h1=self.lrelu(d_bn1(h1,train=train))
            nets.append(h1)
            h1=tf.reshape(h1, [self.batch_size, -1])
            h1=tf.concat([h1,label], 1)

            d_bn2=batch_norm_linear(name='d_bn2')
            h2=linear(h1, dfc_dim, 'd_h2_lin')
            h2=self.lrelu(d_bn2(h2, train=train))
            nets.append(h2)
            h2=tf.concat([h2, label], 1)

            h3=linear(h2, 1, 'd_h3_lin')

            # d_logit=h3
            # output=tf.nn.sigmoid(d_logit)
            return h3, nets

    # pass generated image to measurment model
    def measurement_fn(self, input, name="measurement_fn"):
        with tf.variable_scope(name) as scope:
            if self.measurement == "block_pixels":
                return block_pixels(input, probability=self.prob)
            elif self.measurement == "block_patch":
                return block_patch(input, patch_size=self.patch_size)
            elif self.measurement == "keep_patch":
                return keep_patch(input, patch_size=self.patch_size)
            elif self.measurement == "conv_noise":
                return conv_noise(input, kernel_size=self.kernel_size, stddev=self.stddev)
            # begin added by rick
            elif self.measurement == "block_pixels_patch":
                return block_pixels_patch(input, probability=self.prob, patch_size=self.patch_size)
            # end added by rick
