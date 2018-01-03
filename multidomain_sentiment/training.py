# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import chainer
import numpy
import six
from chainer import cuda


def convert(batch, device):
    def to_device_batch_seq(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    def to_device_batch(batch):
        if device is None:
            return numpy.array(batch)
        elif device < 0:
            batch = numpy.array(batch)
            return chainer.dataset.to_device(device, batch)
        else:
            xp = cuda.cupy.get_array_module(*batch)
            return xp.array(batch)

    keys = list(six.iterkeys(batch[0]))
    return {'xs': to_device_batch_seq([b['xs'] for b in batch]),
            'ys': to_device_batch([b['ys'] for b in batch]),
            'domains': to_device_batch([b['domains'] for b in batch])}
