from tensorflow.keras import models
from models.architectures.blocks import DownsampleBlock
from tensorflow.keras.layers import Input, ZeroPadding2D, BatchNormalization, LeakyReLU, Conv2D

class Discriminator(models.Sequential):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.input = Input(shape=[256, 256, 3])

        self.down1 = DownsampleBlock(64, 4, False)
        self.down2 = DownsampleBlock(128, 4)
        self.down3 = DownsampleBlock(256, 4)

        self.zero_pad1 = ZeroPadding2D()
        self.conv = Conv2D(512, 4, strides=1, use_bias=False)
        self.batchnorm1 = BatchNormalization()
        self.leaky_relu = LeakyReLU()
        self.zero_pad2 = ZeroPadding2D()
        self.last = Conv2D(1, 4, strides=1)

    def call(self, inputs):
        # x = self.input(inputs)
        x = inputs
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)
        x = self.last(x)
        return x
