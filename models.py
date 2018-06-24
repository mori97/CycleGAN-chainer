import chainer
import chainer.functions as F
import chainer.links as L


class Discriminator(chainer.Chain):
    """Discriminator of Cycle-GAN. (70x70 PatchGAN)

    Note: Batch normalization will be instance normalization
          if batch size is 1.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        init_w = chainer.initializers.Normal(scale=0.02)
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, ksize=4, stride=2, pad=1,
                                         initialW=init_w)

            self.conv2 = L.Convolution2D(64, 128, ksize=4, stride=2, pad=1,
                                         initialW=init_w)
            self.inorm2 = L.BatchNormalization(128)

            self.conv3 = L.Convolution2D(128, 256, ksize=4, stride=2, pad=1,
                                         initialW=init_w)
            self.inorm3 = L.BatchNormalization(256)

            self.conv4 = L.Convolution2D(256, 512, ksize=4, stride=2, pad=1,
                                         initialW=init_w)
            self.inorm4 = L.BatchNormalization(512)

            self.conv_out = L.Convolution2D(512, 1, ksize=4, stride=1, pad=1,
                                            initialW=init_w)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.inorm2(self.conv2(h)))
        h = F.leaky_relu(self.inorm3(self.conv3(h)))
        h = F.leaky_relu(self.inorm4(self.conv4(h)))
        return F.average(F.sigmoid(self.conv_out(h)), axis=(1, 2, 3))


class ResNetBlock(chainer.Chain):
    """A building block of ResNet.
    """
    def __init__(self, n_channels, init_w=None):
        super(ResNetBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_channels, n_channels,
                                         pad=1, ksize=3, initialW=init_w)
            self.bnorm1 = L.BatchNormalization(n_channels)
            self.conv2 = L.Convolution2D(n_channels, n_channels,
                                         pad=1, ksize=3, initialW=init_w)
            self.bnorm2 = L.BatchNormalization(n_channels)

    def __call__(self, x):
        h = F.relu(self.bnorm1(self.conv1(x)))
        h = self.bnorm2(self.conv2(h))
        return h + x


class Generator(chainer.Chain):
    """Generator of Cycle-GAN.

    Note: Batch normalization will be instance normalization
    if batch size is 1.
    """
    U64_PAD_WIDTH = ((0, 0), (0, 0), (0, 1), (0, 1))
    U32_PAD_WIDTH = ((0, 0), (0, 0), (1, 0), (1, 0))

    def __init__(self, n_blocks):
        super(Generator, self).__init__()
        init_w = chainer.initializers.Normal(scale=0.02)
        with self.init_scope():
            self.c7s1_32_conv = L.Convolution2D(3, 32, ksize=7,
                                                stride=1, pad=3,
                                                initialW=init_w)
            self.c7s1_32_inorm = L.BatchNormalization(32)
            self.d64_conv = L.Convolution2D(32, 64, ksize=3, stride=2, pad=1,
                                            initialW=init_w)
            self.d64_inorm = L.BatchNormalization(64)
            self.d128_conv = L.Convolution2D(64, 128, ksize=3, stride=2, pad=1,
                                             initialW=init_w)
            self.d128_inorm = L.BatchNormalization(128)
            self.r_blocks = ResNetBlock(128, init_w).repeat(n_blocks)
            self.u64_dconv = L.Deconvolution2D(128, 64, ksize=3,
                                               stride=2, pad=1,
                                               initialW=init_w)
            self.u64_inorm = L.BatchNormalization(64)
            self.u32_dconv = L.Deconvolution2D(64, 32, ksize=3,
                                               stride=2, pad=1,
                                               initialW=init_w)
            self.u32_inorm = L.BatchNormalization(32)
            self.c7s1_3_conv = L.Convolution2D(32, 3, ksize=7, stride=1, pad=3,
                                               initialW=init_w)
            self.c7s1_3_inorm = L.BatchNormalization(3)

    def __call__(self, x):
        h = F.relu(self.c7s1_32_inorm(self.c7s1_32_conv(x)))
        h = F.relu(self.d64_inorm(self.d64_conv(h)))
        h = F.relu(self.d128_inorm(self.d128_conv(h)))
        h = self.r_blocks(h)

        h = self.u64_dconv(h)
        h = F.pad(h, self.U64_PAD_WIDTH, 'constant', constant_values=0)
        h = F.relu(self.u64_inorm(h))
        h = self.u32_dconv(h)
        h = F.pad(h, self.U32_PAD_WIDTH, 'constant', constant_values=0)
        h = F.relu(self.u32_inorm(h))

        h = F.relu(self.c7s1_3_inorm(self.c7s1_3_conv(h)))
        return h
