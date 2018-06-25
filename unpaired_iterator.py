from chainer.dataset import iterator
from chainer.iterators import SerialIterator


class UnpairedIterator(iterator.Iterator):
    """An iterator for unpaired dataset which wraps two SerialIterator.
    """
    def __init__(self, dataset1, dataset2, batch_size, repeat=True):
        if len(dataset2) < len(dataset1):
            self._main_iter = SerialIterator(dataset1, batch_size=batch_size,
                                             repeat=repeat, shuffle=True)
            self._sub_iter = SerialIterator(dataset2, batch_size=batch_size,
                                            repeat=True, shuffle=True)
            self._rev = False
        else:
            self._main_iter = SerialIterator(dataset2, batch_size=batch_size,
                                             repeat=repeat, shuffle=True)
            self._sub_iter = SerialIterator(dataset1, batch_size=batch_size,
                                            repeat=True, shuffle=True)
            self._rev = True

    def __next__(self):
        if self._rev:
            return [x for x in zip(self._sub_iter.next(),
                                   self._main_iter.next())]
        else:
            return [x for x in zip(self._main_iter.next(),
                                   self._sub_iter.next())]

    next = __next__

    @property
    def epoch(self):
        return self._main_iter.epoch

    @property
    def epoch_detail(self):
        return self._main_iter.epoch_detail

    @property
    def previous_epoch_detail(self):
        return self._main_iter.previous_epoch_detail

    def reset(self):
        self._main_iter.reset()
        self._sub_iter.reset()

    @property
    def repeat(self):
        return self._main_iter.repeat

    @property
    def is_new_epoch(self):
        return self._main_iter.is_new_epoch
