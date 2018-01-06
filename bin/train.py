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
from multidomain_sentiment.dataset.blitzer import prepare_blitzer_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.argument('word2vec', type=click.Path(exists=True))
@click.option('--epoch', '-e', type=int, default=15,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', type=int, default=500,
              help='Frequency of taking a snapshot (in iterations)')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--model', default='cnn', type=click.Choice(["cnn", "rnn"]))
@click.option('--batchsize', '-b', type=int, default=300,
              help='Number of images in each mini-batch')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--fix_embedding', type=bool, default=False,
              help='Fix word embedding during training')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(dataset, word2vec, epoch, frequency, gpu, out, model, batchsize, lr,
        fix_embedding, resume):
    """
    Train multi-domain user review classification using Blitzer et al.'s dataset
    (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/)

    Please refer README.md for details.
    """
    memory = Memory(cachedir=out, verbose=1)
    w2v, vocab, train_dataset, dev_dataset, _, label_dict, domain_dict = \
        memory.cache(prepare_blitzer_data)(dataset, word2vec)
    if model == 'rnn':
        model = multidomain_sentiment.models.create_rnn_predictor(
            len(domain_dict), w2v.shape[0], w2v.shape[1], 300, len(label_dict),
            2, 300, dropout_rnn=0.1, initialEmb=w2v, dropout_emb=0.1,
            fix_embedding=fix_embedding
        )
    elif model == 'cnn':
        model = multidomain_sentiment.models.create_cnn_predictor(
            len(domain_dict), w2v.shape[0], w2v.shape[1], 300, len(label_dict),
            300, dropout_fc=0.1, initialEmb=w2v, dropout_emb=0.1,
            fix_embedding=fix_embedding
        )
    else:
        assert not "should not get here"

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
