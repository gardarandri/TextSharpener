

import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageSharpener:
    def __init__(self):
        self.W = []
        self.b = []
        self.conv_info = []

        self.batch_size = 8
        self.num_iterations = 10000
        self.learning_rate = 0.001
        self.reg_const = 1e-9
        self.winit = 1.0

        self.net_image = tf.placeholder(tf.float32, shape = [None,100,100,3])
        self.net_label = tf.placeholder(tf.float32, shape = [None,100,100,3])

        self.sharpened_image = self.init_net(self.net_image)# + self.net_image

        self.stop_criterion = "file"



    def leaky_relu(self, x):
        return tf.maximum(0.1*x,x)

    def init_net(self, input_tensor):
        layer = input_tensor

        layer = tf.reshape(layer, [8,100,100,3])
        
        layer = self.add_conv_layer(layer, [5,5,3,32], 2)
        layer = self.add_conv_layer(layer, [3,3,32,64], 2)
        layer = self.add_conv_layer(layer, [1,1,64,64], 1)
        layer = self.add_conv_layer(layer, [3,3,64,64], 1)

        layer = self.pop_conv_layer(layer)
        layer = self.pop_conv_layer(layer)
        layer = self.pop_conv_layer(layer)
        layer = self.pop_conv_layer(layer)

        layer = self.conv_layer_and_weights(layer, [3,3,3,3], 1, "SAME", tf.nn.relu)


        return layer

    def add_conv_layer(self, x, filter_shape, stride, activation=None):
        if activation == None:
            activation = self.leaky_relu

        input_size = x.get_shape().as_list()
        W = tf.Variable(tf.random_normal(filter_shape, stddev=self.winit / math.sqrt(reduce(lambda x,y: x*y,filter_shape,1.0))))
        x = tf.nn.conv2d(x, W, [1,stride,stride,1], padding="SAME")

        self.conv_info.append((W,input_size,stride))

        self.W.append(W)

        return activation(x)

    def pop_conv_layer(self, x, activation=None):
        if activation == None:
            activation = self.leaky_relu

        info = self.conv_info.pop()

        print(info[1])
        W = tf.Variable(tf.random_normal(info[0].get_shape().as_list(), stddev=self.winit / math.sqrt(reduce(lambda x,y: x*y,info[0].get_shape().as_list(),1.0))))
        x = tf.nn.conv2d_transpose(x, W, tf.stack(info[1]), [1,info[2],info[2],1], padding="SAME")

        self.W.append(W)

        return activation(x)

    def res_layer(self, x, filter_size):
        tmp = self.conv_layer_and_weights(x, [filter_size, filter_size, self.m_channels, self.m_channels], 1, "SAME", tf.nn.relu)
        return tmp + x

    def conv_layer_and_weights(self, x, conv_dims, stride, padding, activation):
        #mean, variance = tf.nn.moments(x, [0])
        #x = tf.nn.batch_normalization(x, mean, variance, tf.Variable(tf.random_normal(mean.get_shape().as_list())), tf.Variable(tf.random_normal(mean.get_shape().as_list())), 0.01)

        W = tf.Variable(tf.random_normal(conv_dims,stddev=1.0/(math.sqrt(sum(conv_dims)))))
        b = tf.Variable(tf.random_normal([1],stddev=1.0/(math.sqrt(sum(conv_dims)))))
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

    def save_sanity_check(self, im_lab, sess, iteration):
        B = min(3,im_lab[0].shape[0])
        # Sanity check
        for i in range(B):
            plt.subplot(B,3,i*3 + 1)
            plt.imshow(im_lab[0][i])
            plt.subplot(B,3,i*3 + 2)
            plt.imshow(im_lab[1][i])
            plt.subplot(B,3,i*3 + 3)
            plt.imshow(sess.run(self.sharpened_image, feed_dict = {
            self.net_image : im_lab[0]
            })[i])
        plt.savefig("iter"+str(iteration)+".png")

    def train_on_images(self, train_files, label_files, val_train_files, val_label_files):
        image, label = self.make_file_pipeline(train_files, label_files)
        val_image, val_label = self.make_file_pipeline(val_train_files, val_label_files)


        cost = tf.reduce_mean((self.sharpened_image - self.net_label)**2)
        #reg = tf.reduce_sum(self.W[0]*self.W[0])
        #for w in self.W[1:]:
        #    reg = reg + tf.reduce_mean(w*w)
        
        train_cost = cost# + self.reg_const*reg

        #train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(train_cost)
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(train_cost)

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        writer = tf.summary.FileWriter("../logs/", sess.graph)
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        vl_cost = []
        tr_cost = []
        tr_iter = []

        #for iteration in range(self.num_iterations):
        iteration = 0
        while True:
            if self.stop_criterion == "iteration" and iteration >= self.num_iterations:
                break

            if self.stop_criterion == "file":
                try:
                    open("stopfile")
                    break
                except IOError:
                    None


            im_lab = sess.run([image,label])

            print(self.conv_info)
            a = sess.run(self.W[-1])
            print(a[0,0,0,0])
            sess.run(train, feed_dict = {
                self.net_image : im_lab[0],
                self.net_label : im_lab[1]
                })
            #b = sess.run(self.W)
            #prstr = ""
            #for i in range(len(self.W)):
            #    prstr = prstr + " " + str(i) + ": "
            #    prstr = prstr + str(np.linalg.norm(a[i] - b[i]))

            #print(prstr)

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
            if iteration % 50 == 0:
                self.save_sanity_check(im_lab, sess, iteration)

            iteration += 1
        
        coord.request_stop()
        coord.join(threads)

        plt.subplot(1,1,1)
        plt.plot(tr_iter, tr_cost)
        plt.plot(tr_iter, vl_cost)
        plt.savefig("train_plot.png")


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

#ims.sharpen(
#        ["../data/validation_1_train"+str(i)+".png" for i in range(10)], "_before"
#        )

ims.train_on_images(
        ["../data/set_1_train"+str(i)+".png" for i in range(1000)],
        ["../data/set_1_label"+str(i)+".png" for i in range(1000)],
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)],
        ["../data/validation_1_label"+str(i)+".png" for i in range(10)]
        )

#ims.sharpen(
#        ["../data/validation_1_train"+str(i)+".png" for i in range(10)], "_after"
#        )


