from models.architectures.blocks import DownsampleBlock, UpsampleBlock
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2DTranspose, Concatenate

class Generator(models.Sequential):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_stack = [
            DownsampleBlock(64, 4, apply_batchnorm=False),
            DownsampleBlock(128, 4),
            DownsampleBlock(256, 4),
            DownsampleBlock(512, 4),
            DownsampleBlock(512, 4),
            DownsampleBlock(512, 4),
            DownsampleBlock(512, 4),
            DownsampleBlock(512, 4),
        ]
        self.up_stack = [
            UpsampleBlock(512, 4, apply_dropout=True),
            UpsampleBlock(512, 4, apply_dropout=True),
            UpsampleBlock(512, 4, apply_dropout=True),
            UpsampleBlock(512, 4),
            UpsampleBlock(256, 4),
            UpsampleBlock(128, 4),
            UpsampleBlock(64, 4),
        ]

        self.last = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        self.concatente = Concatenate()

    def call(self, inputs):
        x = inputs
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = self.concatente([x, skip])
        x = self.last(x)

        return x
