"""TensorFlow library."""
import tensorflow as tf
import numpy as np
import collections

class _TensorFlowNeuralNetwork(object):

    def __init__(self, *args, **kwargs):

        self.architecture = kwargs.pop('architecture')
        self.dropout = kwargs.pop('dropout', False)

        # TensorFlow placeholders.
        self.x = tf.placeholder(tf.float32, [None,
                                             self.architecture[0].shape[1]])
        self.y_ = tf.placeholder(tf.float32, [None,
                                              self.architecture[-1].shape[1]])
        self.keep_prob = tf.placeholder(tf.float32)

        self.weights, self.out = self._initialise_model()

        self.accuracy = _accuracy(self.out, self.y_)

    def _initialise_weights(self):
        raise ValueError('_initialise_weights has not been implemented.')


class VanillaNeuralNetwork(_TensorFlowNeuralNetwork):

    def __init__(self, *args, **kwargs):

        self.learning_rate = kwargs.pop('learning_rate', 0.05)
        self.optimizer = kwargs.pop('optimizer', tf.train.AdamOptimizer)

        super(VanillaNeuralNetwork, self).__init__(*args, **kwargs)

        cost = _cross_entropy(self.out, self.y_)
        self.train_step = self.optimizer(self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialise_model(self):
        architecture = self.architecture
        weights = dict()
        layer = self.x
        for i, (in_, out_) in enumerate(zip(architecture[:-1], architecture[1:])):
            w, b = 'w' + str(i), 'b' + str(i)
            if isinstance(out_, MaxPoolingLayer):
                # Max pooling layers don't have weights.
                weights[w] = None
                weights[b] = None
            elif isinstance(out_, FullyConnectedLayer):
                weights[w] = tf.Variable(tf.random_normal(out_.shape))
                weights[b] = tf.Variable(tf.zeros([out_.shape[1]], dtype=tf.float32))
            elif isinstance(out_, ConvolutionalLayer):
                weights[w] = tf.Variable(tf.truncated_normal(out_.shape, stddev=0.1))
                weights[b] = tf.Variable(tf.constant(0.1, shape=[out_.shape[-1]]))
            else:
                raise ValueError('Must be instance')

            layer = out_.instantiate(layer, weights[w], weights[b], in_,
                                     self.keep_prob)

        return weights, layer


class _TensorFlowLayer(object):

    def __init__(self, *args, **kwargs):
        pass

    def instantiate(self, x, W, b, prev, keep_prob=1.0):
        raise ValueError('instantiate not implemented.')


class FullyConnectedLayer(_TensorFlowLayer):
    
    def __init__(self, shape, nonlin=tf.nn.relu):
        self.shape = shape
        self.nonlin = nonlin

    def instantiate(self, x, W, b, prev, keep_prob=1.0):
        if isinstance(prev, FullyConnectedLayer) or isinstance(prev, MaxPoolingLayer):
            x = tf.reshape(x, [-1, self.shape[0]])

        layer = _linear(x, W, b)
        if self.nonlin: layer = self.nonlin(layer)
        layer = tf.nn.dropout(layer, keep_prob)
        return layer


class MaxPoolingLayer(_TensorFlowLayer):

    def __init__(self, k=2, padding='SAME'):
        self.ksize = [1, k, k, 1]
        self.strides = [1, k, k, 1]
        self.padding = padding

    def instantiate(self, x, W, b, prev, keep_prob=1.0):
        layer = tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return tf.nn.dropout(layer, keep_prob)


class ConvolutionalLayer(_TensorFlowLayer):
    
    def __init__(self, shape, reshape, strides=1, padding='SAME', nonlin=tf.nn.relu):
        self.shape = shape # [width, height, n_in, n_out]
        self.reshape = reshape
        self.strides = [1, strides, strides, 1]
        self.padding = padding
        self.nonlin = nonlin

    def instantiate(self, x, W, b, prev, keep_prob=1.0):
        x = tf.reshape(x, self.reshape)
        x = tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)
        layer = self.nonlin(x) + b
        layer = tf.nn.dropout(layer, keep_prob)
        return layer

def _cross_entropy(pred, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, targets))

def _accuracy(pred, targets):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def _linear(x, w, b):
    return tf.matmul(x, w) + b
