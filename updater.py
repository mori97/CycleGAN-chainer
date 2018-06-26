import chainer
from chainer.backends import cuda
import chainer.functions as F


class CycleGANUpdater(chainer.training.updaters.StandardUpdater):
    """An Updater for Cycle-GAN.
    """
    def __init__(self, train_iter, optimizer, device,
                 lambda_v=10, pool_size=48):
        self._lambda = lambda_v
        super(CycleGANUpdater, self).__init__(
            train_iter, optimizer, device=device)

        # Check if batch size is 1
        # We assume the batch size is 1 in our codes
        batch = train_iter.next()
        if len(batch) != 1:
            raise ValueError('Batch size is not 1.')

        # Make initial fake image pool
        x_images = []
        y_images = []
        # Get pool_size images from each domain
        for _ in range(pool_size):
            batch = cuda.cupy.array(train_iter.next(), dtype=cuda.cupy.float32)
            x_images.append(cuda.cupy.expand_dims(batch[0][0], axis=0))
            y_images.append(cuda.cupy.expand_dims(batch[0][1], axis=0))
        g_gen = optimizer['g_gen'].target
        f_gen = optimizer['f_gen'].target
        # Use batch size 1 when generate fake images to reduce
        # GPU memory usage.
        x_fakes = [f_gen(img).data for img in y_images]
        y_fakes = [g_gen(img).data for img in x_images]
        self._x_fake_pool = cuda.cupy.concatenate(x_fakes, axis=0)
        self._y_fake_pool = cuda.cupy.concatenate(y_fakes, axis=0)

        train_iter.reset()

    def dis_adv_loss(self, dis, real, fake):
        y = dis(real)
        loss = F.mean_squared_error(
            y, cuda.cupy.ones(y.shape, dtype=cuda.cupy.float32))
        y = dis(fake)
        loss += F.mean_squared_error(
            y, cuda.cupy.zeros(y.shape, dtype=cuda.cupy.float32))
        chainer.reporter.report({'dis_loss': loss}, dis)
        return loss / 2

    def gen_adv_loss(self, dis, gen, fake):
        y = dis(fake)
        loss = F.mean_squared_error(
            y, cuda.cupy.ones(y.shape, dtype=cuda.cupy.float32))
        chainer.reporter.report({'gen_loss': loss}, gen)
        return loss

    def update_core(self):
        opt_x_dis = self.get_optimizer('x_dis')
        opt_y_dis = self.get_optimizer('y_dis')
        opt_g_gen = self.get_optimizer('g_gen')
        opt_f_gen = self.get_optimizer('f_gen')

        x_dis = opt_x_dis.target
        y_dis = opt_y_dis.target
        g_gen = opt_g_gen.target
        f_gen = opt_f_gen.target

        batch = self.get_iterator('main').next()
        x = cuda.cupy.array([s[0] for s in batch], dtype=cuda.cupy.float32)
        y = cuda.cupy.array([s[1] for s in batch], dtype=cuda.cupy.float32)
        fake_x = f_gen(y)
        fake_y = g_gen(x)

        # Update the fake pool
        self._x_fake_pool = cuda.cupy.concatenate(
            (self._x_fake_pool[1:, :, :, :], fake_x.data), axis=0)
        self._y_fake_pool = cuda.cupy.concatenate(
            (self._y_fake_pool[1:, :, :, :], fake_y.data), axis=0)

        # Adversarial loss (Discriminator)
        opt_x_dis.update(self.dis_adv_loss, x_dis, x, self._x_fake_pool)
        opt_y_dis.update(self.dis_adv_loss, y_dis, y, self._y_fake_pool)

        # Adversarial loss (Generator)
        opt_g_gen.update(self.gen_adv_loss, y_dis, g_gen, fake_y)
        opt_f_gen.update(self.gen_adv_loss, x_dis, f_gen, fake_x)

        # Cycle consistency loss
        g_gen.cleargrads()
        f_gen.cleargrads()
        cyc_loss_x = F.mean_absolute_error(f_gen(g_gen(x)), x)
        cyc_loss_y = F.mean_absolute_error(g_gen(f_gen(y)), y)
        chainer.reporter.report({'cyc_loss': cyc_loss_x}, g_gen)
        chainer.reporter.report({'cyc_loss': cyc_loss_y}, f_gen)
        cyc_loss = self._lambda * (cyc_loss_x + cyc_loss_y)
        cyc_loss.backward()
        opt_g_gen.update()
        opt_f_gen.update()
