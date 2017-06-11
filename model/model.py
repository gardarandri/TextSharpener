

import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class ImageSharpener:
    def __init__(self):
        self.W = []
        self.b = []

        self.batch_size = 8
        self.num_iterations = 10000
        self.learning_rate = 1.0
        self.reg_const = 0.0

        self.net_image = tf.placeholder(tf.float32, shape = [None,100,100,3])
        self.net_label = tf.placeholder(tf.float32, shape = [None,100,100,3])

        self.sharpened_image = self.init_net(self.net_image) + self.net_image


    def leaky_relu(self, x):
        return tf.maximum(0.03*x,x)

    def init_net(self, input_tensor):
        layer = input_tensor

        layer = self.conv_layer_and_weights(layer, [3,3,3,6], 1, "SAME", self.leaky_relu)
        layer = self.conv_layer_and_weights(layer, [3,3,6,6], 1, "SAME", self.leaky_relu)
        layer = self.conv_layer_and_weights(layer, [3,3,6,6], 1, "SAME", self.leaky_relu)
        layer = self.conv_layer_and_weights(layer, [3,3,6,6], 1, "SAME", self.leaky_relu)
        layer = self.conv_layer_and_weights(layer, [3,3,6,6], 1, "SAME", self.leaky_relu)
        layer = self.conv_layer_and_weights(layer, [3,3,6,3], 1, "SAME", tf.nn.sigmoid)

        return layer

    def res_layer(self, x, filter_size):
        tmp = self.conv_layer_and_weights(x, [filter_size, filter_size, self.m_channels, self.m_channels], 1, "SAME", tf.nn.relu)
        return tmp + x

    def conv_layer_and_weights(self, x, conv_dims, stride, padding, activation):
        mean, variance = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, variance, tf.Variable(tf.random_normal(mean.get_shape().as_list())), tf.Variable(tf.random_normal(mean.get_shape().as_list())), 0.01)

        W = tf.Variable(tf.random_normal(conv_dims,stddev=1e-5/(math.sqrt(sum(conv_dims)))))
        b = tf.Variable(tf.random_normal([1],stddev=1e-5/(math.sqrt(sum(conv_dims)))))
        self.W.append(W)
        self.b.append(b)

        return activation(tf.add(tf.nn.conv2d(x, W, [1,stride,stride,1], padding=padding), b))

    def conv_layer(self,x,W,b,activation):
        res = tf.add(tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME"),b)
        return activation(res)

    def de_conv_layer(self, x,W,b,activation, targ_shape):
        return self.conv_layer(tf.image.resize_images(x,targ_shape),W,b,activation)

    def make_file_pipeline(self, image_files, label_files, batch_size = None):
        if batch_size == None:
            batch_size = self.batch_size

        image_files_prod = tf.train.string_input_producer(image_files, shuffle = True, seed = 1)
        label_files_prod = tf.train.string_input_producer(label_files, shuffle = True, seed = 1)

        reader = tf.WholeFileReader()

        image_file, image = reader.read(image_files_prod)
        label_file, label = reader.read(label_files_prod)

        image = tf.to_float(tf.image.decode_png(image, channels = 3)) / 256.0
        label = tf.to_float(tf.image.decode_png(label, channels = 3)) / 256.0

        image = tf.reshape(image,[100,100,3])
        label = tf.reshape(label,[100,100,3])

        image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size, capacity = 1000)

        return image_batch, label_batch

    def train_on_images(self, train_files, label_files, val_train_files, val_label_files):
        image, label = self.make_file_pipeline(train_files, label_files)
        val_image, val_label = self.make_file_pipeline(val_train_files, val_label_files, 10)


        cost = tf.reduce_mean((self.sharpened_image - self.net_label)**2)
        reg = tf.reduce_sum(self.W[0]*self.W[0])
        for w in self.W[1:]:
            reg = reg + tf.reduce_mean(w*w)
        
        train_cost = cost + self.reg_const*reg

        train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(train_cost)

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        vl_cost = []
        tr_cost = []
        tr_iter = []

        for iteration in range(self.num_iterations):
            im_lab = sess.run([image,label])

            a = sess.run(self.W[-1])
            sess.run(train, feed_dict = {
                self.net_image : im_lab[0],
                self.net_label : im_lab[1]
                })
            b = sess.run(self.W[-1])
            print(np.linalg.norm(a-b))
            if iteration % 5 == 0:
                val_im_lab = sess.run([val_image, val_label])
                vl_cost.append(sess.run(cost, feed_dict = {
                    self.net_image : val_im_lab[0],
                    self.net_label : val_im_lab[1]
                    }))
                tr_cost.append(sess.run(cost, feed_dict = {
                    self.net_image : im_lab[0],
                    self.net_label : im_lab[1]
                    }))
                tr_iter.append(iteration)
                print("Validation cost at iteration {} is {}".format(
                    iteration,
                    vl_cost[-1]
                    ))
                print("Training cost at iteration {} is {}".format(
                    iteration,
                    tr_cost[-1]
                    ))

        im_lab = sess.run([image,label])
        # Sanity check
        plt.subplot(2,2,1)
        plt.imshow(im_lab[0][0])
        plt.subplot(2,2,2)
        plt.imshow(im_lab[1][0])
        plt.subplot(2,2,3)
        plt.imshow(sess.run(self.sharpened_image, feed_dict = {
        self.net_image : im_lab[0]
        })[0])
        plt.show()

        coord.request_stop()
        coord.join(threads)

        plt.plot(tr_iter, tr_cost)
        plt.plot(tr_iter, vl_cost)
        plt.show()


    def sharpen(self, train_files, name_suffix=""):
        image, label = self.make_file_pipeline(train_files, train_files,1)

        image_out = tf.image.encode_png(tf.reshape(tf.cast((self.sharpened_image)*256.0,tf.uint8),[100,100,3]))

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for f_str in train_files:
            out_file = open(f_str[:-4]+"_sharpened"+name_suffix+".png","w")
            out_file.write(sess.run(image_out, feed_dict = {
                self.net_image : sess.run(image)
                }))
            out_file.close()

        coord.request_stop()
        coord.join(threads)
            




ims = ImageSharpener()

tf.set_random_seed(5)

#ims.test()

ims.sharpen(
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)], "_before"
        )

ims.train_on_images(
        ["../data/set_1_train"+str(i)+".png" for i in range(1000)],
        ["../data/set_1_label"+str(i)+".png" for i in range(1000)],
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)],
        ["../data/validation_1_label"+str(i)+".png" for i in range(10)]
        )

ims.sharpen(
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)], "_after"
        )

