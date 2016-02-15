#!/usr/bin/env python3
from statistics import mean

import tensorflow as tf
import numpy as np

from tfx.logging import LogMessage


class BaseModel:
    def __init__(self, data, FLAGS):
        self.data = data
        self.FLAGS = FLAGS

        self.batch_idx = tf.placeholder("int32", name='batch_idx')
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.dropout_keep_prob = tf.placeholder("float32", name='dropout_keep_prob')
        self.use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')


class ModelW2W(BaseModel):
    def __init__(self, data, FLAGS):
        super(ModelW2W, self).__init__(data, FLAGS)

    @staticmethod
    def batch_evaluate(func, indexes):
        tps, tls, tas = [], [], []
        for batch_idx in indexes:
            predictions, lss, acc = func(batch_idx)

            # print('X1', predictions.shape)

            tps.append(np.expand_dims(predictions, axis=0))
            tls.append(float(lss))
            tas.append(float(acc))
        predictions = np.concatenate(tps)
        # print('X1', predictions.shape)
        lss = mean(tls)
        acc = mean(tas)

        return predictions, lss, acc

    def evaluate(self, epoch, learning_rate, sess):
        m = LogMessage()
        m.add()
        m.add('Epoch: {epoch}'.format(epoch=epoch))
        m.add('  - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))


        def batch_eval_1(batch_idx):
            return sess.run(
                [self.predictions, self.loss, self.accuracy],
                feed_dict={
                    self.batch_idx: batch_idx,
                    self.use_inputs_prob: 1.0,
                    self.dropout_keep_prob: 1.0,
                    self.phase_train: False,
                }
            )

        train_predictions, train_loss, train_accuracy_action = self.batch_evaluate(batch_eval_1, self.data.train_batch_indexes)

        m.add('  Train data')
        m.add('    - use inputs prob = {uip:f}'.format(uip=1.0))
        m.add('      - loss          = {lss:f}'.format(lss=train_loss))
        m.add('      - accuracy      = {acc:f}'.format(acc=train_accuracy_action))

        def batch_eval_0(batch_idx):
            return sess.run(
                [self.predictions, self.loss, self.accuracy],
                feed_dict={
                    self.batch_idx: batch_idx,
                    self.use_inputs_prob: 0.0,
                    self.dropout_keep_prob: 1.0,
                    self.phase_train: False,
                }
            )

        self.train_predictions_action, train_loss, train_accuracy_action = self.batch_evaluate(batch_eval_0, self.data.train_batch_indexes)

        self.train_predictions_action_argmax = np.argmax(self.train_predictions_action, axis=2)

        m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
        m.add('      - loss          = {lss:f}'.format(lss=train_loss))
        m.add('      - accuracy      = {acc:f}'.format(acc=train_accuracy_action))

        self.dev_predictions_action, dev_loss, dev_accuracy_action = self.batch_evaluate(batch_eval_0, self.data.dev_batch_indexes)

        self.dev_predictions_action_argmax = np.argmax(self.dev_predictions_action, axis=2)

        m.add('  Dev data')
        m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
        m.add('      - loss          = {lss:f}'.format(lss=dev_loss))
        m.add('      - accuracy      = {acc:f}'.format(acc=dev_accuracy_action))
        m.add()
        m.log()

        self.test_predictions_action, test_loss, test_accuracy_action = self.batch_evaluate(batch_eval_0, self.data.test_batch_indexes)

        self.test_predictions_action_argmax = np.argmax(self.test_predictions_action, axis=2)

        m.add('  Test data')
        m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
        m.add('      - loss          = {lss:f}'.format(lss=test_loss))
        m.add('      - accuracy      = {acc:f}'.format(acc=test_accuracy_action))
        m.add()
        m.log()

        return train_accuracy_action, train_loss, \
               dev_accuracy_action, dev_loss, \
               test_accuracy_action, test_loss


    def log_predictions_dataset(self, log_fn, actions, batch_indexes):
        m = LogMessage(log_fn=log_fn)
        m.add('Shape of action predictions: {s}'.format(s=actions.shape))
        m.add('Argmax predictions')
        m.add()

        for prediction_batch_idx, batch_idx in enumerate(batch_indexes):
            for history in range(0, self.data.batch_histories.shape[1]):
                m.add('History {h}'.format(h=prediction_batch_idx * self.FLAGS.batch_size + history))

                for j in range(self.data.batch_histories.shape[2]):
                    utterance = []
                    for k in range(self.data.batch_histories.shape[3]):
                        w = self.data.idx2word_history[self.data.batch_histories[batch_idx, history, j, k]]
                        if w not in ['_SOS_', '_EOS_']:
                            utterance.append(w)
                    if utterance:
                        m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

                prediction = []
                for j in range(actions.shape[2]):
                    w = self.data.idx2word_action[actions[prediction_batch_idx, history, j]]
                    if w not in ['_SOS_', '_EOS_']:
                        prediction.append(w)

                m.add('P  : {t:80}'.format(t=' '.join(prediction)))

                target = []
                for j in range(self.data.batch_actions.shape[2]):
                    w = self.data.idx2word_action[self.data.batch_actions[batch_idx, history, j]]
                    if w not in ['_SOS_', '_EOS_']:
                        target.append(w)

                m.add('T  : {t:80}'.format(t=' '.join(target)))
                m.add()
        m.log(print_console=False)

    def log_predictions(self):
        self.log_predictions_dataset(
            'predictions_train_set.txt',
            actions=self.train_predictions_action_argmax,
            batch_indexes=self.data.train_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_dev_set.txt',
            actions=self.dev_predictions_action_argmax,
            batch_indexes=self.data.dev_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_test_set.txt',
            actions=self.test_predictions_action_argmax,
            batch_indexes=self.data.test_batch_indexes
        )

class ModelW2T(BaseModel):
    def __init__(self, data, FLAGS):
        super(ModelW2T, self).__init__(data, FLAGS)

    @staticmethod
    def batch_evaluate(func, indexes):
        tps, tls, tas = [], [], []
        for batch_idx in indexes:
            predictions, lss, acc = func(batch_idx)

            # print('X1', predictions.shape)

            tps.append(np.expand_dims(predictions, axis=0))
            tls.append(float(lss))
            tas.append(float(acc))
        predictions = np.concatenate(tps)
        # print('X1', predictions.shape)
        lss = mean(tls)
        acc = mean(tas)

        return predictions, lss, acc

    def evaluate(self, epoch, learning_rate, sess):
        m = LogMessage()
        m.add('')
        m.add('Epoch: {epoch}'.format(epoch=epoch))
        m.add('  - learning rate   = {lr:e}'.format(lr=learning_rate.eval()))


        def batch_eval(batch_idx):
            return sess.run(
                [self.predictions, self.loss, self.accuracy],
                feed_dict={
                    self.batch_idx: batch_idx,
                    self.use_inputs_prob: 1.0,
                    self.dropout_keep_prob: 1.0,
                    self.phase_train: False,
                }
            )

        self.train_predictions_action, train_loss, train_accuracy_action = self.batch_evaluate(batch_eval, self.data.train_batch_indexes)

        self.train_predictions_action_argmax = np.argmax(self.train_predictions_action, axis=2)

        m.add('  Train data')
        m.add('    - loss          = {lss:f}'.format(lss=train_loss))
        m.add('    - accuracy      = {acc:f}'.format(acc=train_accuracy_action))

        self.dev_predictions_action, dev_loss, dev_accuracy_action = self.batch_evaluate(batch_eval, self.data.dev_batch_indexes)

        self.dev_predictions_action_argmax = np.argmax(self.dev_predictions_action, axis=2)

        m.add('  Dev data')
        m.add('    - loss          = {lss:f}'.format(lss=dev_loss))
        m.add('    - accuracy      = {acc:f}'.format(acc=dev_accuracy_action))

        self.test_predictions_action, test_loss, test_accuracy_action = self.batch_evaluate(batch_eval, self.data.test_batch_indexes)

        self.test_predictions_action_argmax = np.argmax(self.test_predictions_action, axis=2)

        m.add('  Test data')
        m.add('    - loss          = {lss:f}'.format(lss=test_loss))
        m.add('    - accuracy      = {acc:f}'.format(acc=test_accuracy_action))
        m.add()
        m.log()

        return train_accuracy_action, train_loss, \
               dev_accuracy_action, dev_loss, \
               test_accuracy_action, test_loss


    def log_predictions_dataset(self, log_fn, actions_template, batch_indexes):
        m = LogMessage(log_fn=log_fn)
        m.add('Shape of action template predictions: {s}'.format(s=actions_template.shape))
        m.add()
        m.add('Predictions')
        m.add()

        # print(self.data.batch_histories.shape)
        # print(self.data.batch_actions_template.shape)
        # print(actions_template.shape)
        # print(len(batch_indexes))
        for prediction_batch_idx, batch_idx in enumerate(batch_indexes):
            for history in range(0, self.data.batch_histories.shape[1]):
                m.add('History {h}'.format(h=prediction_batch_idx * self.FLAGS.batch_size + history))

                for j in range(self.data.batch_histories.shape[2]):
                    utterance = []
                    for k in range(self.data.batch_histories.shape[3]):
                        w = self.data.idx2word_history[self.data.batch_histories[batch_idx, history, j, k]]
                        if w not in ['_SOS_', '_EOS_']:
                            utterance.append(w)
                    if utterance:
                        m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

                m.add('P  : {t:80}'.format(
                    t=self.data.idx2word_action_template[actions_template[prediction_batch_idx, history]])
                )
                m.add('T  : {t:80}'.format(
                    t=self.data.idx2word_action_template[self.data.batch_actions_template[batch_idx, history]])
                )
                m.add()
                # m.log()
        m.log(print_console=False)

    def log_predictions(self):
        self.log_predictions_dataset(
            'predictions_train_set.txt',
            actions_template=self.train_predictions_action_argmax,
            batch_indexes=self.data.train_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_dev_set.txt',
            actions_template=self.dev_predictions_action_argmax,
            batch_indexes=self.data.dev_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_test_set.txt',
            actions_template=self.test_predictions_action_argmax,
            batch_indexes=self.data.test_batch_indexes
        )

class ModelW2TArgs(BaseModel):
    def __init__(self, data, FLAGS):
        super(ModelW2TArgs, self).__init__(data, FLAGS)

    @staticmethod
    def batch_evaluate(func, indexes):
        tp1s, tp2s, tls, ta1s, ta2s = [], [], [], [], []
        for batch_idx in indexes:
            predictions1, predictions2, lss, acc1, acc2 = func(batch_idx)

            # print('X1', predictions.shape)

            tp1s.append(np.expand_dims(predictions1, axis=0))
            tp2s.append(np.expand_dims(predictions2, axis=0))
            tls.append(float(lss))
            ta1s.append(float(acc1))
            ta2s.append(float(acc2))
        predictions1 = np.concatenate(tp1s)
        predictions2 = np.concatenate(tp2s)
        # print('X1', predictions.shape)
        lss = mean(tls)
        acc1 = mean(ta1s)
        acc2 = mean(ta2s)

        return predictions1, predictions2, lss, acc1, acc2

    def evaluate(self, epoch, learning_rate, sess):
        m = LogMessage()
        m.add('')
        m.add('Epoch: {epoch}'.format(epoch=epoch))
        m.add('  - learning rate   = {lr:e}'.format(lr=learning_rate.eval()))

        def batch_eval(batch_idx):
            return sess.run(
                [
                    self.predictions_action, self.predictions_arguments,
                    self.loss,
                    self.accuracy_action, self.accuracy_arguments
                ],
                feed_dict={
                    self.batch_idx: batch_idx,
                    self.use_inputs_prob: 1.0,
                    self.dropout_keep_prob: 1.0,
                    self.phase_train: False,
                }
            )

        self.train_predictions_action, self.train_predictions_arguments, \
        train_loss, \
        train_accuracy_action, train_accuracy_arguments = self.batch_evaluate(batch_eval, self.data.train_batch_indexes)

        self.train_predictions_action_argmax = np.argmax(self.train_predictions_action, axis=2)
        self.train_predictions_arguments_argmax = np.argmax(self.train_predictions_arguments, axis=3)

        m.add('  Train data')
        m.add('    - loss               = {lss:f}'.format(lss=train_loss))
        m.add('    - accuracy action    = {acc:f}'.format(acc=train_accuracy_action))
        m.add('    - accuracy arguments = {acc:f}'.format(acc=train_accuracy_arguments))

        self.dev_predictions_action, self.dev_predictions_arguments, \
        dev_loss, \
        dev_accuracy_action, dev_accuracy_arguments = self.batch_evaluate(batch_eval, self.data.dev_batch_indexes)

        self.dev_predictions_action_argmax = np.argmax(self.dev_predictions_action, axis=2)
        self.dev_predictions_arguments_argmax = np.argmax(self.dev_predictions_arguments, axis=3)

        m.add('  Dev data')
        m.add('    - loss               = {lss:f}'.format(lss=dev_loss))
        m.add('    - accuracy action    = {acc:f}'.format(acc=dev_accuracy_action))
        m.add('    - accuracy arguments = {acc:f}'.format(acc=dev_accuracy_arguments))

        self.test_predictions_action, self.test_predictions_arguments, \
        test_loss, \
        test_accuracy_action, test_accuracy_arguments = self.batch_evaluate(batch_eval, self.data.test_batch_indexes)

        self.test_predictions_action_argmax = np.argmax(self.test_predictions_action, axis=2)
        self.test_predictions_arguments_argmax = np.argmax(self.test_predictions_arguments, axis=3)

        m.add('  Test data')
        m.add('    - loss               = {lss:f}'.format(lss=test_loss))
        m.add('    - accuracy action    = {acc:f}'.format(acc=test_accuracy_action))
        m.add('    - accuracy arguments = {acc:f}'.format(acc=test_accuracy_arguments))
        m.add()
        m.log()

        return 0.5 * (train_accuracy_action + train_accuracy_arguments), train_loss, \
               0.5 * (dev_accuracy_action + dev_accuracy_arguments), dev_loss, \
               0.5 * (test_accuracy_action + test_accuracy_arguments), test_loss

    def log_predictions_dataset(self, log_fn, actions_template, actions_arguments, batch_indexes):
        m = LogMessage(log_fn=log_fn)
        m.add('Shape of action template predictions: {s}'.format(s=actions_template.shape))
        m.add('Shape of action arguments predictions: {s}'.format(s=actions_arguments.shape))
        m.add()
        m.add('Predictions')
        m.add()

        # print(self.data.batch_histories.shape)
        # print(self.data.batch_actions_template.shape)
        # print(self.data.batch_actions_arguments.shape)
        # print(actions_template.shape)
        # print(actions_arguments.shape)
        # print(len(batch_indexes))

        for prediction_batch_idx, batch_idx in enumerate(batch_indexes):
            for history in range(0, self.data.batch_histories.shape[1]):
                m.add('History {h}'.format(h=prediction_batch_idx * self.FLAGS.batch_size + history))

                for j in range(self.data.batch_histories.shape[2]):
                    utterance = []
                    for k in range(self.data.batch_histories.shape[3]):
                        w = self.data.idx2word_history[self.data.batch_histories[batch_idx, history, j, k]]
                        if w not in ['_SOS_', '_EOS_']:
                            utterance.append(w)
                    if utterance:
                        m.add('U {j}  : {c:80}'.format(j=j, c=' '.join(utterance)))

                m.add('P    : {t:80}'.format(
                    t=self.data.idx2word_action_template[actions_template[prediction_batch_idx, history]])
                )

                w_actions_arguments = []
                for j in range(actions_arguments.shape[2]):
                    w = self.data.idx2word_action_arguments[actions_arguments[prediction_batch_idx, history, j]]
                    w_actions_arguments.append(w)

                m.add('ArgsP: {t:80}'.format(t=', '.join(w_actions_arguments)))

                m.add('T    : {t:80}'.format(
                    t=self.data.idx2word_action_template[self.data.batch_actions_template[batch_idx, history]])
                )

                w_actions_arguments = []
                for j in range(self.data.batch_actions_arguments.shape[2]):
                    w = self.data.idx2word_action_arguments[self.data.batch_actions_arguments[batch_idx, history, j]]
                    w_actions_arguments.append(w)

                m.add('ArgsT: {t:80}'.format(t=', '.join(w_actions_arguments)))

                m.add()
                # m.log()
        m.log(print_console=False)

    def log_predictions(self):
        self.log_predictions_dataset(
            'predictions_train_set.txt',
            actions_template=self.train_predictions_action_argmax,
            actions_arguments=self.train_predictions_arguments_argmax,
            batch_indexes=self.data.train_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_dev_set.txt',
            actions_template=self.dev_predictions_action_argmax,
            actions_arguments=self.dev_predictions_arguments_argmax,
            batch_indexes=self.data.dev_batch_indexes
        )
        self.log_predictions_dataset(
            'predictions_test_set.txt',
            actions_template=self.test_predictions_action_argmax,
            actions_arguments=self.test_predictions_arguments_argmax,
            batch_indexes=self.data.test_batch_indexes
        )
