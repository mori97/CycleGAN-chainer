import argparse
import os

import chainer
from chainer import iterators, optimizers, training
from chainer.backends import cuda
from chainer.training import extensions
import numpy as np
from PIL import Image

from models import Discriminator, Generator
from unpaired_iterator import UnpairedIterator
from updater import CycleGANUpdater


def output_fake_images(g_gen, f_gen, test_a_iter, test_b_iter, dst_dir):
    """"""
    @training.make_extension()
    def _output_fake_images(trainer):
        dst_dir_fake_a = os.path.join(dst_dir, 'fakeA')
        dst_dir_fake_b = os.path.join(dst_dir, 'fakeB')
        if not os.path.exists(dst_dir_fake_a):
            os.makedirs(dst_dir_fake_a)
        if not os.path.exists(dst_dir_fake_b):
            os.makedirs(dst_dir_fake_b)

        for batch in test_a_iter:
            filenames = [s[0] for s in batch]
            x = cuda.cupy.array([s[1] for s in batch], dtype=cuda.cupy.float32)
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    y = g_gen(x)
            y = y.data.transpose((0, 2, 3, 1))
            y *= 255
            y = cuda.to_cpu(y).astype(np.uint8)
            for filename, img_array in zip(filenames, y):
                dst_path = os.path.join(dst_dir, 'fakeB', filename)
                Image.fromarray(img_array).save(dst_path)

        for batch in test_b_iter:
            filenames = [s[0] for s in batch]
            x = cuda.cupy.array([s[1] for s in batch], dtype=cuda.cupy.float32)
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    y = f_gen(x)
            y = y.data.transpose((0, 2, 3, 1))
            y *= 255
            y = cuda.to_cpu(y).astype(np.uint8)
            for filename, img_array in zip(filenames, y):
                dst_path = os.path.join(dst_dir, 'fakeA', filename)
                Image.fromarray(img_array).save(dst_path)

        test_a_iter.reset()
        test_b_iter.reset()

    return _output_fake_images


def make_dataset_iterator(d_path):
    """Make dataset iterators.

    :param d_path: path of the dataset which should have a 'trainA' folder,
                 a 'testA' folder, a 'trainB' folder and a 'testB' folder.
    :type d_path: str

    :return: A iterator of training set and a iterator of test set.
    :rtype: tuple[Chainer.iterators.SerialIterator]
    """
    exts = ('.jpg', '.png')

    def _convert_image_to_array(file_path):
        img = Image.open(file_path)
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255
        img_array = img_array.transpose((2, 0, 1))
        return img_array

    dataset_train_a = []
    for filename in os.listdir(os.path.join(d_path, 'trainA')):
        if os.path.splitext(filename)[1] in exts:
            dataset_train_a.append(_convert_image_to_array(
                os.path.join(d_path, 'trainA', filename)))

    dataset_train_b = []
    for filename in os.listdir(os.path.join(d_path, 'trainB')):
        if os.path.splitext(filename)[1] in exts:
            dataset_train_b.append(_convert_image_to_array(
                os.path.join(d_path, 'trainB', filename)))

    dataset_test_a = []
    for filename in os.listdir(os.path.join(d_path, 'testA')):
        if os.path.splitext(filename)[1] in exts:
            dataset_test_a.append(
                (filename,
                 _convert_image_to_array(
                     os.path.join(d_path, 'testA', filename))))

    dataset_test_b = []
    for filename in os.listdir(os.path.join(d_path, 'testB')):
        if os.path.splitext(filename)[1] in exts:
            dataset_test_b.append(
                (filename,
                 _convert_image_to_array(
                     os.path.join(d_path, 'testB', filename))))

    train_iter = UnpairedIterator(dataset_train_a, dataset_train_b,
                                  batch_size=1, repeat=True)
    test_a_iter = iterators.SerialIterator(dataset_test_a, batch_size=1,
                                           repeat=False, shuffle=False)
    test_b_iter = iterators.SerialIterator(dataset_test_b, batch_size=1,
                                           repeat=False, shuffle=False)
    return train_iter, test_a_iter, test_b_iter


def main():
    parser = argparse.ArgumentParser(prog='Train Cycle-GAN.')
    parser.add_argument('--dataset-path',
                        help='',
                        type=str, required=True)
    parser.add_argument('--device',
                        help='',
                        type=int, default=0)
    parser.add_argument('--epochs',
                        help='',
                        type=int, default=100)
    parser.add_argument('--n-blocks',
                        help='',
                        type=int, default=9)
    parser.add_argument('--out',
                        help='',
                        type=str, default='./result')
    args = parser.parse_args()

    x_dis = Discriminator()
    y_dis = Discriminator()
    g_gen = Generator(n_blocks=args.n_blocks)
    f_gen = Generator(n_blocks=args.n_blocks)
    if args.device >= 0:
        cuda.get_device_from_id(args.device).use()
        x_dis.to_gpu()
        y_dis.to_gpu()
        g_gen.to_gpu()
        f_gen.to_gpu()

    opt_x_dis = optimizers.Adam(0.0002)
    opt_x_dis.setup(x_dis)
    opt_y_dis = optimizers.Adam(0.0002)
    opt_y_dis.setup(y_dis)
    opt_g_gen = optimizers.Adam(0.0002)
    opt_g_gen.setup(g_gen)
    opt_f_gen = optimizers.Adam(0.0002)
    opt_f_gen.setup(f_gen)

    train_iter, test_a_iter, test_b_iter =\
        make_dataset_iterator(args.dataset_path)

    updater = CycleGANUpdater(
        train_iter=train_iter,
        optimizer={'x_dis': opt_x_dis, 'y_dis': opt_y_dis,
                   'g_gen': opt_g_gen, 'f_gen': opt_f_gen},
        device=args.device)

    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.LogReport())
    trainer.extend(output_fake_images(g_gen, f_gen,
                                      test_a_iter, test_b_iter,
                                      args.out),
                   trigger=(2, 'epoch'))
    trainer.run()

    chainer.serializers.save_hdf5('x_dis.hdf5', x_dis)
    chainer.serializers.save_hdf5('y_dis.hdf5', y_dis)
    chainer.serializers.save_hdf5('g_gen.hdf5', g_gen)
    chainer.serializers.save_hdf5('f_gen.hdf5', f_gen)


if __name__ == '__main__':
    main()