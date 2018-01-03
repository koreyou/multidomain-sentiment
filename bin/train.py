# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import json
import logging
import os

import chainer
import dill  # This is for joblib to use dill. Do NOT delete it.
import click
import six
from chainer import training
from chainer.training import extensions
from joblib import Memory

import multidomain_sentiment
from multidomain_sentiment.dataset import prepare_amazon_review_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--epoch', '-e', type=int, default=15,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', type=int, default=500,
              help='Frequency of taking a snapshot (in iterations)')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--word2vec', required=True, type=click.Path(exists=True),
              help='Word2vec pretrained file path')
@click.option('--train', '-t', multiple=True,
              type=(six.text_type, click.Path(exists=True)),
              help='Pairs of domain:path')
@click.option('--dev', '-d', multiple=True,
              type=(six.text_type, click.Path(exists=True)),
              help='Pairs of domain:path')
@click.option('--batchsize', '-b', type=int, default=50,
              help='Number of images in each mini-batch')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--fix_embedding', type=bool, default=False,
              help='Fix word embedding during training')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(epoch, frequency, gpu, out, word2vec, train, dev,
        batchsize, lr, fix_embedding, resume):
    memory = Memory(cachedir=out, verbose=1)
    w2v, vocab, train_dataset, dev_dataset, label_dict, domain_dict = \
        memory.cache(prepare_amazon_review_dataset)(train, dev, word2vec)
    model = multidomain_sentiment.models.create_multi_domain_predictor(
        len(domain_dict), w2v.shape[0], w2v.shape[1], 300, len(label_dict),
        2, 300, dropout_rnn=0.1, initialEmb=w2v, dropout_emb=0.1,
        fix_embedding=fix_embedding
    )
    classifier = multidomain_sentiment.models.MultiDomainClassifier(
        model, domain_dict=domain_dict)

    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        classifier.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(classifier)

    train_iter = chainer.iterators.SerialIterator(train_dataset, batchsize)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu,
        converter=multidomain_sentiment.training.convert)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    if dev_dataset is not None:
        logger.info("train: {},  dev: {}".format(
            len(train_dataset), len(dev_dataset)))
        # Evaluate the model with the development dataset for each epoch
        dev_iter = chainer.iterators.SerialIterator(
            dev_dataset, batchsize, repeat=False, shuffle=False)

        evaluator = extensions.Evaluator(
            dev_iter, model, device=gpu,
            converter=multidomain_sentiment.training.convert)
        trainer.extend(evaluator, trigger=(1, 'epoch'))
    else:
        logger.info("train: {}".format(len(train_dataset)))

    logger.info("With labels: %s" % json.dumps(label_dict))
    # Take a snapshot for each specified epoch
    trigger = (epoch, 'epoch') if frequency == -1 else (frequency, 'iteration')
    trainer.extend(extensions.snapshot(), trigger=trigger)
    if gpu < 0:
        # ParameterStatistics does not work with GPU as of chainer 2.x
        # https://github.com/chainer/chainer/issues/3027
        trainer.extend(extensions.ParameterStatistics(
            model, trigger=(10, 'iteration')))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()

    # Save final model (without trainer)
    model.save(os.path.join(out, 'trained_model'))
    with open(os.path.join(out, 'vocab.json'), 'wb') as fout:
        json.dump(vocab, fout)


if __name__ == '__main__':
    run()