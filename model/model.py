

import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImageSharpener:
    def __init__(self):
        self.W = {}
        self.W_cnt = 0
        self.b = []
        self.conv_info = []

        self.batch_size = 8
        self.num_iterations = 1000
        self.learning_rate = 0.003
        self.reg_const = 1e-9
        self.winit = 1.0

        self.net_image = tf.placeholder(tf.float32, shape = [None,100,100,3])
        self.net_label = tf.placeholder(tf.float32, shape = [None,100,100,3])

        self.large_image = tf.placeholder(tf.float32, shape = [None,1000,1000,3])

        self.sharpened_image = self.init_net(self.net_image)

        self.stop_criterion = "file"
        self.save = True



    def leaky_relu(self, x):
        return tf.maximum(0.1*x,x)

    def init_net(self, input_tensor, batch_size = None, im_width=100, im_height=100):
        layer = input_tensor

        if batch_size == None:
            batch_size = self.batch_size

        layer = tf.reshape(layer, [batch_size,im_width,im_height,3])
        

        layer = self.conv_layer_and_weights(layer, [3,3,3,6], 1, "SAME", tf.nn.relu, wkey="K1")
        layer = self.conv_layer_and_weights(layer, [3,3,6,12], 1, "SAME", tf.nn.relu, wkey="K2")
        layer = self.conv_layer_and_weights(layer, [3,3,12,12], 1, "SAME", tf.nn.relu, wkey="K3")
        layer = self.conv_layer_and_weights(layer, [3,3,12,6], 1, "SAME", tf.nn.relu, wkey="K4")
        layer = self.conv_layer_and_weights(layer, [3,3,6,3], 1, "SAME", tf.nn.relu, wkey="K5")

        return layer

    def add_variable(self, V, wkey=None):
        W = None
        if wkey in self.W:
            W = self.W[wkey]
        else:
            W = tf.Variable(V)
            if wkey==None:
                wkey = str(self.W_cnt)
                self.W_cnt += 1
            self.W[wkey] = W
        return W

    def add_conv_layer(self, x, filter_shape, stride, activation=None, batch_size=None, wkey=None):
        if activation == None:
            activation = self.leaky_relu
        if batch_size == None:
            batch_size = self.batch_size

        input_size = x.get_shape().as_list()

        W = self.add_variable(tf.random_normal(filter_shape, stddev=self.winit / math.sqrt(reduce(lambda x,y: x*y,filter_shape,1.0))), wkey)
        x = tf.nn.conv2d(x, W, [1,stride,stride,1], padding="SAME")

        self.conv_info.append((W,input_size,stride))

        return activation(x)

    def pop_conv_layer(self, x, activation=None, batch_size=None, wkey=None):
        if activation == None:
            activation = self.leaky_relu
        if batch_size == None:
            batch_size = self.batch_size

        info = self.conv_info.pop()

        W = self.add_variable(tf.random_normal(info[0].get_shape().as_list(), stddev=self.winit / math.sqrt(reduce(lambda x,y: x*y,info[0].get_shape().as_list(),1.0))), wkey)
        x = tf.nn.conv2d_transpose(x, W, tf.stack(info[1]), [1,info[2],info[2],1], padding="SAME")

        return activation(x)

    def conv_layer_and_weights(self, x, conv_dims, stride, padding, activation, wkey=None):
        #mean, variance = tf.nn.moments(x, [0])
        #x = tf.nn.batch_normalization(x, mean, variance, tf.Variable(tf.random_normal(mean.get_shape().as_list())), tf.Variable(tf.random_normal(mean.get_shape().as_list())), 0.01)

        W = self.add_variable(tf.random_normal(conv_dims,stddev=1.0/(math.sqrt(sum(conv_dims)))), wkey)
        b = None
        if wkey != None:
            b = self.add_variable(tf.random_normal([1],stddev=1.0/(math.sqrt(sum(conv_dims)))), wkey+"_bias")
        else:
            b = self.add_variable(tf.random_normal([1],stddev=1.0/(math.sqrt(sum(conv_dims)))))

        return activation(tf.add(tf.nn.conv2d(x, W, [1,stride,stride,1], padding=padding), b))

    #def res_layer(self, x, filter_size):
    #    tmp = self.conv_layer_and_weights(x, [filter_size, filter_size, self.m_channels, self.m_channels], 1, "SAME", tf.nn.relu)
    #    return tmp + x


    #def conv_layer(self,x,W,b,activation):
    #    res = tf.add(tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME"),b)
    #    return activation(res)

    #def de_conv_layer(self, x,W,b,activation, targ_shape):
    #    return self.conv_layer(tf.image.resize_images(x,targ_shape),W,b,activation)

    def make_file_pipeline(self, image_files, label_files, batch_size = None, im_width=100, im_height=100, shuffle=True):
        if batch_size == None:
            batch_size = self.batch_size

        image_files_prod = tf.train.string_input_producer(image_files, shuffle = shuffle, seed = 1)
        label_files_prod = tf.train.string_input_producer(label_files, shuffle = shuffle, seed = 1)

        reader = tf.WholeFileReader()

        image_file, image = reader.read(image_files_prod)
        label_file, label = reader.read(label_files_prod)

        image = tf.to_float(tf.image.decode_png(image, channels = 3)) / 256.0
        label = tf.to_float(tf.image.decode_png(label, channels = 3)) / 256.0

        image = tf.reshape(image,[im_width,im_height,3])
        label = tf.reshape(label,[im_width,im_height,3])

        image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size, capacity = 1000)

        return image_batch, label_batch

    def save_sanity_check(self, im_lab, sess, iteration):
        B = min(3,im_lab[0].shape[0])
        # Sanity check
        for i in range(B):
            plt.subplot(B,3,i*3 + 1)
            plt.imshow(im_lab[0][i])
            plt.axis("off")
            plt.subplot(B,3,i*3 + 2)
            plt.imshow(im_lab[1][i])
            plt.axis("off")
            plt.subplot(B,3,i*3 + 3)
            plt.imshow(sess.run(self.sharpened_image, feed_dict = {
            self.net_image : im_lab[0]
            })[i])
            plt.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("iter"+str(iteration)+".png", bbox_inches="tight")

    def start_session(self):
        sess = tf.Session()
        return sess

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

        sess = self.start_session()
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
                    # If stopfile exists then we stop training
                    open("stopfile")
                    break
                except IOError:
                    None


            im_lab = sess.run([image,label])

            a = sess.run(self.W["K1"])
            print(a[0,0,0,0])
            sess.run(train, feed_dict = {
                self.net_image : im_lab[0],
                self.net_label : im_lab[1]
                })

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

        if self.save == True:
            saver = tf.train.Saver()
            saver.save(sess,"../savedmodels/sharpener")

        return sess

    def load_model(self, model_path):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        return sess

    def sharpen(self, train_files, name_suffix="", sess = None):
        self.sharpened_image = self.init_net(self.net_image,1)

        image, label = self.make_file_pipeline(train_files, train_files,1)

        image_out = tf.image.encode_png(tf.reshape(tf.cast((self.sharpened_image)*256.0,tf.uint8),[100,100,3]))

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        if sess == None:
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

    def test(self, sess):
        train_files = ["../data/large_train"+str(i)+".png" for i in range(10)]

        self.sharpened_image = self.init_net(self.large_image,1,1000,1000)

        image, label = self.make_file_pipeline(train_files,train_files,1,im_width=1000,im_height=1000, shuffle=False)

        image_out = tf.image.encode_png(tf.reshape(tf.cast((self.sharpened_image)*256.0,tf.uint8),[1000,1000,3]))

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        if sess == None:
            sess = tf.Session()
            sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for f_str in train_files:
            out_file = open(f_str[:-4]+"_sharpened.png","w")
            out_file.write(sess.run(image_out, feed_dict = {
                self.large_image : sess.run(image)
                }))
            out_file.close()

        coord.request_stop()
        coord.join(threads)

    def diagnostics(self, sess):
        train_files = ["../data/large_train"+str(i)+".png" for i in range(10)]

        self.sharpened_image = self.init_net(self.large_image,1,1000,1000)

        image, label = self.make_file_pipeline(train_files,train_files,1,im_width=1000,im_height=1000, shuffle=False)

        image_out = tf.image.encode_png(tf.reshape(tf.cast((self.sharpened_image)*256.0,tf.uint8),[1000,1000,3]))

        init_op  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        if sess == None:
            sess = tf.Session()
            sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        eim = sess.run(self.encoded_image, feed_dict = {
            self.large_image : sess.run(image)
            })

        plt.imshow(eim[0][:,:,0:3])
        plt.show()

        coord.request_stop()
        coord.join(threads)

ims = ImageSharpener()

tf.set_random_seed(5)

sess = ims.train_on_images(
        ["../data/set_1_train"+str(i)+".png" for i in range(1000)],
        ["../data/set_1_label"+str(i)+".png" for i in range(1000)],
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)],
        ["../data/validation_1_label"+str(i)+".png" for i in range(10)]
        )

sess = ims.load_model("../savedmodels/sharpener")

ims.sharpen(
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)], "_after",
        sess
        )

#ims.test(sess)
#ims.diagnostics(sess)

