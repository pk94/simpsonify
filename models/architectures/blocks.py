import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, ReLU, Layer

class DownsampleBlock(Layer):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(DownsampleBlock, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(filters, size, strides=2, padding='same', use_bias=False))
        if apply_batchnorm:
            self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

    def call(self, inputs, **kwargs):
        return self.model(inputs)


class UpsampleBlock(Layer):
    def __init__(self, filters, size, apply_dropout=False):
        super(UpsampleBlock, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False))
        self.model.add(BatchNormalization())
        if apply_dropout:
            self.model.add(Dropout(0.5))
        self.model.add(ReLU())

    def call(self, inputs, **kwargs):
        return self.model(inputs)
