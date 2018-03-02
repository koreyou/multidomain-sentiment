import chainer
import chainer.functions as F
from chainer import reporter

from multidomain_sentiment.models.discriminator import Discriminator


class MultiDomainClassifier(chainer.Chain):

    def __init__(self, model, domain_dict=None):
        super(MultiDomainClassifier, self).__init__()
        with self.init_scope():
            self.model = model
            self.domain_dict = domain_dict

    def __call__(self, xs, ys, domains):
        concat_outputs = self.predict(xs, domains)
        loss = F.softmax_cross_entropy(concat_outputs, ys)
        accuracy = F.accuracy(concat_outputs, ys)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        output_cpu = chainer.cuda.to_cpu(concat_outputs.data)
        ys_cpu = chainer.cuda.to_cpu(ys)
        for k in set(domains):
            domain_mask = domains == k
            accuracy = F.accuracy(output_cpu[domain_mask], ys_cpu[domain_mask])
            if self.domain_dict is not None:
                name = 'accuracy_%s' % self.domain_dict[k]
            else:
                name = 'accuracy_%d' % k
            reporter.report({name: accuracy.data}, self)
        return loss

    def predict(self, xs, domains, softmax=False, argmax=False):
        o, _ = self.model(xs, domains)
        if softmax:
            return F.softmax(o).data
        elif argmax:
            return self.xp.argmax(o.data, axis=1)
        else:
            return o


class AdversarialMultiDomainClassifier(chainer.Chain):

    def __init__(self, model, n_domains, domain_dict=None, adv_coeff=0.05):
        super(AdversarialMultiDomainClassifier, self).__init__()
        with self.init_scope():
            self.adv_coeff = adv_coeff
            self.model = model
            self.discriminator = Discriminator(n_domains)
            self.domain_dict = domain_dict

    def __call__(self, xs, ys, domains):
        concat_outputs, shared_features = self.predict(xs, domains)
        discriminator_outputs = self.discriminator(shared_features)
        loss_discriminator = F.softmax_cross_entropy(discriminator_outputs, domains)
        reporter.report({'loss_discriminator': loss_discriminator.data}, self)
        loss_predictor = F.softmax_cross_entropy(concat_outputs, ys)
        accuracy = F.accuracy(concat_outputs, ys)
        reporter.report({'loss_predictor': loss_predictor.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        loss = loss_predictor + self.adv_coeff * loss_discriminator
        reporter.report({'loss': loss.data}, self)
        output_cpu = chainer.cuda.to_cpu(concat_outputs.data)
        ys_cpu = chainer.cuda.to_cpu(ys)
        for k in set(domains):
            domain_mask = domains == k
            accuracy = F.accuracy(output_cpu[domain_mask], ys_cpu[domain_mask])
            if self.domain_dict is not None:
                name = 'accuracy_%s' % self.domain_dict[k]
            else:
                name = 'accuracy_%d' % k
            reporter.report({name: accuracy.data}, self)
        return loss

    def predict(self, xs, domains, softmax=False, argmax=False):
        o, shared_features = self.model(xs, domains)
        if softmax:
            return F.softmax(o).data
        elif argmax:
            return self.xp.argmax(o.data, axis=1), shared_features
        else:
            return o, shared_features
