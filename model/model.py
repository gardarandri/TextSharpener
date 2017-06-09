
import tensorflow as tf
import numpy as np


class ImageSharpener:
    def __init__(self):
        #self.W1 = tf.Variable(tf.random_normal([8,8,3,6],stddev=1/256.0))
        #self.b1 = tf.Variable(tf.random_normal([1]))
        self.W = []
        self.b = []

        self.batch_size = 8


    def init_net(self, input_tensor):
        layer = input_tensor

        #layer = self.conv_layer_and_weights(layer, [3,3,3,10], 1, "SAME", tf.nn.relu)
        #layer = self.conv_layer_and_weights(layer, [3,3,10,10], 1, "SAME", tf.nn.relu)
        #layer = self.conv_layer_and_weights(layer, [3,3,10,10], 1, "SAME", tf.nn.relu)
        #layer = self.conv_layer_and_weights(layer, [3,3,10,3], 1, "SAME", tf.nn.tanh)

        layer = self.conv_layer_and_weights(layer, [3,3,3,10], 1, "SAME", lambda x: tf.maximum(0.2*x,x))
        layer = self.conv_layer_and_weights(layer, [3,3,10,10], 1, "SAME", lambda x: tf.maximum(0.2*x,x))
        layer = self.conv_layer_and_weights(layer, [3,3,10,10], 1, "SAME", lambda x: tf.maximum(0.2*x,x))
        layer = self.conv_layer_and_weights(layer, [3,3,10,10], 1, "SAME", lambda x: tf.maximum(0.2*x,x))
        layer = self.conv_layer_and_weights(layer, [3,3,10,3], 1, "SAME", tf.nn.tanh)

        #self.m_channels = 64
        #layer = self.conv_layer_and_weights(layer, [9,9,3,32], 1, "SAME", tf.nn.relu)
        #s = layer.get_shape().as_list()
        #layer = self.conv_layer_and_weights(layer, [3,3,32,self.m_channels], 1, "SAME", tf.nn.relu)
        #layer = self.res_layer(layer, 3)
        #layer = self.res_layer(layer, 3)
        #layer = self.res_layer(layer, 3)
        #layer = self.res_layer(layer, 3)
        #layer = self.conv_layer_and_weights(layer, [3,3,self.m_channels,32], 1, "SAME", tf.nn.relu)
        #layer = self.conv_layer_and_weights(layer, [3,3,32,3], 1, "SAME", tf.nn.tanh)

        return layer/ 2.0 + 0.5

    def res_layer(self, x, filter_size):
        tmp = self.conv_layer_and_weights(x, [filter_size, filter_size, self.m_channels, self.m_channels], 1, "SAME", tf.nn.relu)
        return tmp + x

    def conv_layer_and_weights(self, x, conv_dims, stride, padding, activation):
        W = tf.Variable(tf.random_normal(conv_dims,stddev=1.0/(sum(conv_dims))))
        b = tf.Variable(tf.random_normal([1],stddev=1.0/(sum(conv_dims))))
        self.W.append(W)
        self.b.append(b)

        return activation(tf.add(tf.nn.conv2d(x, W, [1,stride,stride,1], padding=padding), b))

    def conv_layer(self,x,W,b,activation):
        res = tf.add(tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME"),b)
        return activation(res)

    def de_conv_layer(self, x,W,b,activation, targ_shape):
        return self.conv_layer(tf.image.resize_images(x,targ_shape),W,b,activation)

    def make_file_pipeline(self, train_files, label_files):
        input_queue = tf.train.slice_input_producer([train_files, label_files],shuffle=False)

        image_file = tf.read_file(input_queue[0])
        label_file = tf.read_file(input_queue[1])

        image = tf.to_float(tf.image.decode_png(image_file, channels = 3)) / 256.0
        label = tf.to_float(tf.image.decode_png(label_file, channels = 3)) / 256.0

        image = tf.reshape(image,[100,100,3])
        label = tf.reshape(label,[100,100,3])

        image_batch, label_batch = tf.train.batch([image,label], batch_size = self.batch_size)

        #        training_data_queue = tf.train.string_input_producer(
        #                train_files
        #        , shuffle=True, seed=1)
        #        
        #        training_label_queue = tf.train.string_input_producer(
        #                label_files
        #        , shuffle=True, seed=1)
        #        
        #        data_reader = tf.WholeFileReader()
        #        data_key, data_value = data_reader.read(training_data_queue)
        #        image = tf.image.decode_png(data_value, channels=3)
        #        
        #        label_reader = tf.WholeFileReader()
        #        label_key, label_value = label_reader.read(training_label_queue)
        #        label = tf.image.decode_png(data_value, channels=3)
        #        
        #        image_norm = tf.reshape(tf.to_float(image) / 256.0, [1,100,100,3])
        #        label_norm = tf.reshape(tf.to_float(label) / 265.0, [1,100,100,3])
        #
        #        image_batch = tf.train.batch(image_norm, 8)
        #        label_batch = tf.train.batch(image_norm, 8)
        
        #return tf.train.slice_input_producer([image_batch, label_batch])
        return image_batch, label_batch

    def train_on_images(self, train_files, label_files, val_train_files, val_label_files):
        train_data, train_label = self.make_file_pipeline(train_files, label_files)
        val_data, val_label = self.make_file_pipeline(val_train_files, val_label_files)

        train_net_out = self.init_net(train_data)
        val_net_out = self.init_net(val_data)

        cost = tf.reduce_mean((train_net_out - train_label)**2)
        val_cost = tf.reduce_mean((val_net_out - val_label)**2)

        #cost = tf.reduce_mean((train_data + train_net_out - train_label)**2)
        #val_cost = tf.reduce_mean((val_data + val_net_out - val_label)**2)

        #cost = tf.reduce_mean(2**((train_data + train_net_out - train_label)**2))
        #val_cost = tf.reduce_mean(2**((val_data + val_net_out - val_label)**2))
        
        train = tf.train.AdamOptimizer(0.01).minimize(cost)
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        
            sess.run(tf.global_variables_initializer())
            
            for _ in range(2000):
                a = None
                b = None
                if _ % 10 == 0:
                    print("Cost at iteration {} is {}".format(_,sess.run([cost])[0]))
                    a = sess.run([self.W[0]])

                sess.run([train])

                if _ % 10 == 0:
                    b = sess.run([self.W[0]])
                    print(np.linalg.norm(a[0][0,0]-b[0][0,0]))
        
            cost_sum = 0
            for _ in range(len(val_train_files)/self.batch_size):
                cost_sum += sess.run([val_cost])[0]
            print("Validation cost {}".format(cost_sum / (len(val_train_files) / 8)))

            coord.request_stop()
            coord.join(threads)

    def sharpen(self, train_files):
        train_data, train_label = self.make_file_pipeline(train_files, train_files, 1)
        train_net_out = self.init_net(train_data)

        image_out = tf.image.encode_png(tf.reshape(tf.cast((tf.add(train_data,train_net_out))*256.0,tf.uint8),[100,100,3]))

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        
            sess.run(tf.global_variables_initializer())
            
            for f_str in train_files:
                out_file = open(f_str[:-4]+"_sharpened.png","w")
                out_file.write(sess.run([image_out])[0])
                out_file.close()

            coord.request_stop()
            coord.join(threads)



ims = ImageSharpener()

tf.set_random_seed(5)

ims.train_on_images(
        ["../data/set_1_train"+str(i)+".png" for i in range(1000)],
        ["../data/set_1_label"+str(i)+".png" for i in range(1000)],
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)],
        ["../data/validation_1_label"+str(i)+".png" for i in range(10)]
        )

ims.sharpen(
        ["../data/validation_1_train"+str(i)+".png" for i in range(10)],
        )

