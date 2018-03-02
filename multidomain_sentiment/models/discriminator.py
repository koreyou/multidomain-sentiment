# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer import utils
from chainer.utils import type_check

class GradientReversalLayer(chainer.function_node.FunctionNode):

    """ Gradient Reversal Layer in "Ganin & Lempitsky. 2015. Unsupervised Domain
    Adaptation by Backpropagation. ICML."

    This layer provides very easy way to solve Min-Max optimization in deep
    learning.
    """

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward(self, inputs):
        return inputs[0],

    def backward(self, inputs, grad_outputs):
        return -grad_outputs[0],


def gradient_reversal_layer(x):
    return GradientReversalLayer().apply((x, ))[0]


class Discriminator(chainer.Chain):
    def __init__(self, n_class):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_class)

    def __call__(self, shared_features):
        shared_features = gradient_reversal_layer(shared_features)
        return self.l1(shared_features)
