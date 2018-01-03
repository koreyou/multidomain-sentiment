# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import json
import logging

import nltk
import six

nltk.download(info_or_id='punkt')  # need it before importing nltk.tokenize

import numpy as np
from chainer.datasets import DictDataset
from nltk.tokenize import word_tokenize

from multidomain_sentiment.word_embedding import load_word_embedding

logger = logging.getLogger(__name__)



def create_dataset(texts, labels, domains, size=-1):

    if size > 0:
        # Sample data AFTER all data has been loaded. This is because
        # There might be bias in data ordering.
        ind = np.random.permutation(len(texts))[:size]
        if labels is None:
            return DictDataset(
                xs=[texts[i] for i in ind],
                domains=[domains[i] for i in ind])
        else:
            return DictDataset(
                xs=[texts[i] for i in ind], ys=[labels[i] for i in ind],
                domains=[domains[i] for i in ind])
    else:
        if labels is None:
            return DictDataset(xs=texts, domains=domains)
        else:
            return DictDataset(xs=texts, ys=labels, domains=domains)


def read_amazon_reviews(
        domain_paths, vocab, label_dict, domain_dict=None, max_tokens=10000):
    texts = []
    labels = []
    domains = []
    create_domain = domain_dict is None
    domain_dict = domain_dict or {}
    for domain_name, path in domain_paths:
        if create_domain:
            domain_dict[domain_name] = len(domain_dict)
        d = domain_dict[domain_name]
        for t, l in read_single_review(path, vocab, label_dict, max_tokens):
            texts.append(t)
            labels.append(l)
            domains.append(d)
    labels = np.asarray(labels, np.int32)
    domains = np.asarray(domains, np.int32)
    return create_dataset(texts, labels, domains), domain_dict


def prepare_amazon_review_dataset(train_paths, test_paths, word2vec_path):
    logger.info("Preparing data")

    label_dict, label_inv_dict = get_sentiment_label_dict()

    logger.info("Loading word embedding")
    w2v, vocab = load_word_embedding(word2vec_path, max_vocab=100000)

    logger.info("Creating dataset")
    train, domain_dict = read_amazon_reviews(train_paths, vocab, label_dict, max_tokens=50)

    if len(test_paths) > 0:
        assert len(train_paths) == len(test_paths)
        test, _ = read_amazon_reviews(train_paths, vocab, label_dict,
                                      domain_dict=domain_dict, max_tokens=50)
    else:
        test = None
    # Reverse domain_dict
    domain_dict = {v: k for k, v in six.iteritems(domain_dict)}

    return w2v, vocab, train, test, label_inv_dict, domain_dict


def read_single_review(path, vocab, label_dict, max_tokens):
    with open(path) as fin:
        for line in fin:
            data = json.loads(line.strip())
            label = int(data['overall'])
            if label in label_dict:
                words = []
                for i, w in enumerate(word_tokenize(data['reviewText'])):
                    if i >= max_tokens:
                        break
                    words.append(vocab.get(w, vocab['<unk>']))
                if len(words) == 0:
                    continue
                yield np.array(words, np.int32), label_dict[label]


def get_sentiment_label_dict():
    return ({
        1: 0,
        2: 0,
        3: 0,
        5: 1,
    }, {0: "neg", 1: "pos"})
