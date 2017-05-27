
import tensorflow as tf
import numpy as np


class ImageSharpener:
    def __init__(self):

        self.W1 = tf.Variable(tf.random_normal([8,8,3,6],stddev=1/256.0))
        self.W2 = tf.Variable(tf.random_normal([8,8,6,12],stddev=1/256.0))
        self.W3 = tf.Variable(tf.random_normal([8,8,12,12],stddev=1/256.0))
        self.W4 = tf.Variable(tf.random_normal([8,8,12,6],stddev=1/256.0))
        self.W5 = tf.Variable(tf.random_normal([8,8,6,3],stddev=1/256.0))

        self.W6 = tf.Variable(tf.random_normal([8,8,3,6],stddev=1/256.0))
        self.W7 = tf.Variable(tf.random_normal([8,8,6,6],stddev=1/256.0))
        self.W8 = tf.Variable(tf.random_normal([8,8,6,3],stddev=1/256.0))
        self.W9 = tf.Variable(tf.random_normal([8,8,3,3],stddev=1/256.0))

        self.b1 = tf.Variable(tf.random_normal([1]))
        self.b2 = tf.Variable(tf.random_normal([1]))
        self.b3 = tf.Variable(tf.random_normal([1]))
        self.b4 = tf.Variable(tf.random_normal([1]))
        self.b5 = tf.Variable(tf.random_normal([1]))

        self.b6 = tf.Variable(tf.random_normal([1]))
        self.b7 = tf.Variable(tf.random_normal([1]))
        self.b8 = tf.Variable(tf.random_normal([1]))
        self.b9 = tf.Variable(tf.random_normal([1]))


    def init_net(self, input_tensor):
        print("Model layout:")

        layer = self.conv_layer(input_tensor, self.W1, self.b1, tf.nn.relu)
        print(layer.get_shape())
        
        layer = self.conv_layer(layer, self.W2, self.b2, tf.nn.relu)
        print(layer.get_shape())

        layer = tf.nn.max_pool(layer,[1,8,8,1],[1,2,2,1],padding="VALID")
        print(layer.get_shape())
        
        layer = self.conv_layer(layer, self.W3, self.b3, tf.nn.relu)
        print(layer.get_shape())
        
        layer = self.conv_layer(layer, self.W4, self.b4, tf.nn.relu)
        print(layer.get_shape())

        layer = tf.nn.max_pool(layer,[1,8,8,1],[1,2,2,1],padding="VALID")
        print(layer.get_shape())

        smallest_w = layer.get_shape().as_list()[1]
        smallest_h = layer.get_shape().as_list()[2]

        print("deconv")
        layer = self.de_conv_layer(layer, self.W5, self.b5, tf.nn.relu, [smallest_w*2,smallest_h*2])
        print(layer.get_shape())

        layer = self.de_conv_layer(layer, self.W6, self.b6, tf.nn.relu, [smallest_w*4,smallest_h*4])
        print(layer.get_shape())

        layer = self.de_conv_layer(layer, self.W8, self.b8, tf.nn.relu, [100,100])
        print(layer.get_shape())

        layer = self.conv_layer(layer, self.W9, self.b9, tf.identity)
        print(layer.get_shape())
        print("done!")
        print()
        
        return 2.0*layer - 1.0

    def conv_layer(self,x,W,b,activation):
        res = tf.add(tf.nn.conv2d(x,W,[1,1,1,1],padding="SAME"),b)
        return activation(res)

    def de_conv_layer(self, x,W,b,activation, targ_shape):
        return self.conv_layer(tf.image.resize_images(x,targ_shape),W,b,activation)

    def make_file_pipeline(self, train_files, label_files, batch_size = 16):
        input_queue = tf.train.slice_input_producer([train_files, label_files],shuffle=False)

        image_file = tf.read_file(input_queue[0])
        label_file = tf.read_file(input_queue[1])

        image = tf.to_float(tf.image.decode_png(image_file, channels = 3)) / 256.0
        label = tf.to_float(tf.image.decode_png(label_file, channels = 3)) / 256.0

        image = tf.reshape(image,[100,100,3])
        label = tf.reshape(label,[100,100,3])

        image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size)

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
        
        train = tf.train.GradientDescentOptimizer(0.003).minimize(cost)
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
        
            sess.run(tf.global_variables_initializer())
            
            for _ in range(200):
                if _ % 10 == 0:
                    print("Cost at iteration {} is {}".format(_,sess.run([cost])[0]))
                sess.run([train])
        
            cost_sum = 0
            for _ in range(len(val_train_files)):
                cost_sum += sess.run([val_cost])[0]
            print("Validation cost {}".format(cost_sum / len(val_train_files)))

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

