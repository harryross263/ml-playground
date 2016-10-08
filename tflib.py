"""TensorFlow library."""
import tensorflow as tf
import numpy as np
import collections

class _TensorFlowNeuralNetwork(object):

    def __init__(self, *args, **kwargs):

        self.architecture = kwargs.pop('architecture')

        if not isinstance(self.architecture[0], int):
            raise ValueError('First layer in the network\'s architecture must \
                             be specified as an integer.')
        if not isinstance(self.architecture[-1], int):
            raise ValueError('Last layer in the network\'s architecture must \
                             be specified as an integer.')

        self.x = tf.placeholder(tf.float32, [None, self.architecture[0]])
        self.y_ = tf.placeholder(tf.float32, [None, self.architecture[-1]])

        self.weights = self._initialise_weights()
        self.out = self._generate_model()

        self.accuracy = _accuracy(self.out, self.y_)

    def _initialise_weights(self):
        raise ValueError('_initialise_weights has not been implemented.')

    def _generate_model(self):
        raise ValueError('_generate_model has not been implemented.')
        

class VanillaNeuralNetwork(_TensorFlowNeuralNetwork):

    def __init__(self, *args, **kwargs):

        self.nonlin = kwargs.pop('nonlin', tf.nn.relu)
        self.learning_rate = kwargs.pop('learning_rate', 0.05)
        self.optimizer = kwargs.pop('optimizer', tf.train.AdamOptimizer)

        super(VanillaNeuralNetwork, self).__init__(*args, **kwargs)

        cost = _cross_entropy(self.out, self.y_)
        self.train_step = self.optimizer(self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialise_weights(self):
        architecture = self.architecture
        weights = dict()
        for i, (n_in, n_out) in enumerate(zip(architecture[:-1], architecture[1:])):
            weights['w' + str(i)] = tf.Variable(tf.random_normal([n_in, n_out]))
            weights['b' + str(i)] = tf.Variable(tf.zeros([n_out], dtype=tf.float32))

        return weights

    def _generate_model(self):
        architecture = self.architecture
        nonlin = self.nonlin
        layer = self.x
        for i in range(len(architecture) - 1):
            w, b = self.weights['w' + str(i)], self.weights['b' + str(i)]
            layer = nonlin(_linear(layer, w, b)) if i < len(architecture) - 2  else _linear(layer, w, b)
            
        return layer
                
class _TensorFlowLayer(object):

    def __init__(self, *args, **kwargs):
        pass

    def instantiate(self, prev, W=None, b=None):
        raise ValueError('instance not implemented.')

class FullyConnectedLayer(_TensorFlowLayer):
    
    def __init__(self, dimension=10, nonlin=tf.nn.relu):
        self.dimension = dimension
        self.nonlin = nonlin

    def instantiate(self, x, W, b, prev=None):
        if not prev or isinstance(prev, FullyConnectedLayer):
            return self.nonlin(_linear(x, W, b)) if self.nonlin else _linear(x, W, b)
        elif isinstance(prev, ConvolutionalLayer):
            pass
        elif isinstance(prev, MaxPoolingLayer):
            pass

class MaxPoolingLayer(_TensorFlowLayer):

    def __init__(self, k=2, padding='SAME'):
        self.ksize = [1, k, k, 1]
        self.strides = [1, k, k, 1]
        self.padding = padding

    def instantiate(self, x, W=None, b=None, prev=None):
        if not prev:
            raise ValueError('Invalid architecture: A max-pooling layer cannot \
                             be the first layer.')
        if isinstance(prev, FullyConnectedLayer):
            pass
        elif isinstance(prev, ConvolutionalLayer):
            pass
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

class ConvolutionalLayer(_TensorFlowLayer):
    
    def __init__(self, shape, strides=1, padding='SAME', nonlin=tf.nn.relu):
        self.shape = shape
        self.strides = [1, strides, strides, 1]
        self.padding = padding
        self.nonlin = nonlin

    def instantiate(self, x, W, b):
        x = tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)
        return self.nonlin(tf.nn.bias_add(x, b))

    def weights(self):
        return tf.Variable(tf.truncated_normal(self.shape, stddev=0.1))

    def bias(self):
        return tf.Variable(tf.constant(0.1, shape=self.shape[-1]))

class ConvolutionalNeuralNetwork(_TensorFlowNeuralNetwork):

    def __init__(self, *args, **kwargs):

        self.nonlin = kwargs.pop('nonlin', tf.nn.relu)
        self.learning_rate = kwargs.pop('learning_rate', 0.05)
        self.optimizer = kwargs.pop('optimizer', tf.train.AdamOptimizer)

        super(ConvolutionalNeuralNetwork, self).__init__(architecture)

        cost = _cross_entropy(self.out, self.y_)
        self.train_step = self.optimizer(self.learning_rate).minimize(cost)

        self.accuracy = _accuracy(self.out, self.y_)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialise_weights(self):
        architecture = self.architecture
        weights = dict()
        
        for i, in_, out_ in enumerate(zip(architecture[:-1], architecture[1:])):
            if isinstance(architecture[i], ConvolutionalLayer):
                weights['w_conv' + str(i)] = architecture[i].weights()
                weights['b_conv' + str(i)] = architecture[i].bias()
            elif isinstance(architecture[i], MaxPoolingLayer):
                # A max pooling layer doesn't have any weights.
                pass
            else:
                # Fully connected.
                weights['w' + str(i)] = tf.Variable(tf.random_normal(architecture[i]))
                weights['b' + str(i)] = tf.Variable(tf.zeros([architecture[i][-1]],
                                                             dtype=tf.float32))

        return weights

    def _generate_model(self):
        architecture = self.architecture
        nonlin = self.nonlin
        
        w, b = self.weights['w0'], self.weights['b0']
        layer = nonlin(_linear(self.x, w, b))

        for i in range(1, len(architecture) - 2):
            if isinstance(architecture[i], ConvolutionalLayer):
                w, b = self.weights['w_conv' + str(i)], self.weights['b_conv' + str(i)]
                layer = architecture[i].instantiate(layer, w, b)
            elif isinstance(architecture[i], MaxPoolingLayer):
                layer = architecture[i].instantiate(layer)
            else:
                # Fully connected.
                w, b = self.weights['w' + str(i)], self.weights['b' + str(i)]
                layer = tf.reshape(layer, [-1, tf.shape(w)[0]])
                layer = nonlin(_linear(layer, w, b))

        depth = len(architecture) - 2
        w, b = self.weights['w' + str(depth)], self.weights['b' + str(i)]
        layer = tf.reshape(layer, [-1, tf.shape(w)[0]])
        return _linear(layer, w, b)

def _cross_entropy(pred, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, targets))

def _accuracy(pred, targets):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def _linear(x, w, b):
    return tf.matmul(x, w) + b
