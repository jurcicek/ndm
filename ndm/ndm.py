#!/usr/bin/env python3
import os
import sys
from random import seed, shuffle

import multiprocessing
from statistics import mean, stdev
from time import sleep

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

import dataset
import model_cnn_w2w as cnn_w2w
import model_rnn_w2w as rnn_w2w
import model_cnn12_w2t as cnn12_w2t
import model_cnn12_bn_w2t as cnn12_bn_w2t
import model_cnn12_mp_bn_w2t as cnn12_mp_bn_w2t
import model_cnn12_att_a_w2t as cnn12_att_a_w2t
import model_cnn12_bn_att_a_w2t as cnn12_bn_att_a_w2t
import model_cnn12_bn_att_a_bn_w2t as cnn12_bn_att_a_bn_w2t
import model_cnn12_mp_bn_att_a_w2t as cnn12_mp_bn_att_a_w2t
import model_cnn13_bn_w2t as cnn13_bn_w2t
import model_cnn13_mp_bn_w2t as cnn13_mp_bn_w2t
import model_cnn23_mp_bn_w2t as cnn23_mp_bn_w2t
import model_rnn1_w2t as rnn1_w2t
import model_rnn2_w2t as rnn2_w2t

from tfx.bricks import device_for_node_cpu, device_for_node_gpu, device_for_node_gpu_matmul, \
    device_for_node_gpu_selection
from tfx.optimizers import AdamPlusOptimizer, AdamPlusCovOptimizer
from tfx.logging import start_experiment, LogMessage, LogExperiment
from tfx.various import make_hash

import tfx.logging as logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('runs', 1, 'Number of parallel runs of the trainer.')
flags.DEFINE_integer('threads', 2, 'Number of parallel threads for each run.')
flags.DEFINE_boolean('gpu', False, 'Run the computation on a GPU.')
flags.DEFINE_string('model', 'cnn12-bn-w2t',
                    '"cnn-w2w" (convolutional network for state tracking - words 2 words ) | '
                    '"rnn-w2w" (bidirectional recurrent network for state tracking - words 2 words) | '
                    '"cnn12-w2t" (convolutional network for state tracking - words 2 template | '
                    '"cnn12-bn-w2t" (convolutional network for state tracking - words 2 template | '
                    '"cnn12-mp-bn-w2t" (convolutional network for state tracking - words 2 template | '
                    '"cnn12-att-a-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                    '"cnn12-bn-att-a-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                    '"cnn12-bn-att-a-bn-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                    '"cnn12-mp-bn-att-a-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                    '"cnn13-bn-w2t" (convolutional network for state tracking - words 2 template | '
                    '"cnn13-mp-bn-w2t" (convolutional network for state tracking - words 2 template | '
                    '"cnn23-mp-bn-w2t" (convolutional network for state tracking - words 2 template | '
                    '"rnn1-w2t" (forward only recurrent network for state tracking - words 2 template | '
                    '"rnn2-w2t" (bidirectional recurrent network for state tracking - words 2 template)')
flags.DEFINE_string('task', 'w2t',
                    '"tracker" (dialogue state tracker) | '
                    '"w2w" (word to word dialogue management) | '
                    '"w2t" (word to template dialogue management)')
flags.DEFINE_string('input', 'asr',
                    '"asr" automatically recognised user input | '
                    '"trs" manually transcribed user input | '
                    '"trs+asr" manually transcribed and automatically recognised user input')
flags.DEFINE_string('train_data', './data.dstc2.train.json', 'The train data.')
flags.DEFINE_string('dev_data', './data.dstc2.dev.json', 'The development data.')
flags.DEFINE_string('test_data', './data.dstc2.test.json', 'The test data.')
flags.DEFINE_float('data_fraction', 0.1, 'The fraction of data to usd to train model.')
flags.DEFINE_string('ontology', './data.dstc2.ontology.json', 'The ontology defining slots and their values.')
flags.DEFINE_string('database', './data.dstc2.db.json', 'The backend database defining entries that can be queried.')
flags.DEFINE_integer('max_epochs', 100, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in a batch.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('decay', 0.9, 'AdamPlusOptimizer learning rate decay.')
flags.DEFINE_float('beta1', 0.9, 'AdamPlusOptimizer 1st moment decay.')
flags.DEFINE_float('beta2', 0.999, 'AdamPlusOptimizer 2nd moment decay.')
flags.DEFINE_float('epsilon', 1e-5, 'AdamPlusOptimizer epsilon.')
flags.DEFINE_float('pow', 1.0, 'AdamPlusOptimizer pow.')
flags.DEFINE_float('dense_regularization', 1e-16, 'Weight of regularization for dense updates.')
flags.DEFINE_float('sparse_regularization', 1e-16, 'Weight of regularization foir sparse updates.')
flags.DEFINE_float('max_gradient_norm', 5e0, 'Clip gradients to this norm.')
flags.DEFINE_float('use_inputs_prob_decay', 0.999, 'Decay of the probability of using '
                                                   'the true targets during generation.')
flags.DEFINE_float('dropout_keep_prob', 1.0, '(1 - dropout_keep_prob) is the probability of dropout during training.')
flags.DEFINE_boolean('print_variables', False, 'Print all trainable variables.')

"""
This code shows how to build and train a neural dialogue manager.
There are several models available:
 1) bidirectional RNN for encoding utterances
 2) convolutional neural network for encoding utterances
"""


def log_predictions_w2t(log_fn, model, histories, batch_indexes, predictions_argmax, targets, idx2word_target):
    m = LogMessage(log_fn=log_fn)
    m.add('Shape of predictions: {s}'.format(s=predictions_argmax.shape))
    m.add('Argmax predictions')
    m.add()

    for prediction_batch_idx, batch_idx in enumerate(batch_indexes):
        # print(histories.shape)
        # print(predictions_argmax.shape)
        # print(prediction_batch_idx, batch_idx)
        for history in range(0, histories.shape[1]):
            m.add('History {h}'.format(h=prediction_batch_idx * FLAGS.batch_size + history))

            for j in range(histories.shape[2]):
                utterance = []
                for k in range(histories.shape[3]):
                    w = model.data.idx2word_history[histories[batch_idx, history, j, k]]
                    if w not in ['_SOS_', '_EOS_']:
                        utterance.append(w)
                if utterance:
                    m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

            m.add('P  : {t:80}'.format(t=idx2word_target[predictions_argmax[prediction_batch_idx, history]]))
            m.add('T  : {t:80}'.format(t=idx2word_target[targets[batch_idx, history]]))
            m.add()
            # m.log()
    m.log(print_console=False)


def log_predictions_w2w(log_fn, model, histories, batch_indexes, predictions_argmax, targets, idx2word_target):
    m = LogMessage(log_fn=log_fn)
    m.add('Shape of predictions: {s}'.format(s=predictions_argmax.shape))
    m.add('Argmax predictions')
    m.add()

    for prediction_batch_idx, batch_idx in enumerate(batch_indexes):
        for history in range(0, histories.shape[1]):
            m.add('History {h}'.format(h=prediction_batch_idx * FLAGS.batch_size + history))

            for j in range(histories.shape[2]):
                utterance = []
                for k in range(histories.shape[3]):
                    w = model.data.idx2word_history[histories[batch_idx, history, j, k]]
                    if w not in ['_SOS_', '_EOS_']:
                        utterance.append(w)
                if utterance:
                    m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

            prediction = []
            for j in range(predictions_argmax.shape[2]):
                w = idx2word_target[predictions_argmax[prediction_batch_idx, history, j]]
                if w not in ['_SOS_', '_EOS_']:
                    prediction.append(w)

            m.add('P  : {t:80}'.format(t=' '.join(prediction)))

            target = []
            for j in range(targets.shape[2]):
                w = idx2word_target[targets[batch_idx, history, j]]
                if w not in ['_SOS_', '_EOS_']:
                    target.append(w)

            m.add('T  : {t:80}'.format(t=' '.join(target)))
            m.add()
    m.log(print_console=False)


def batch_evaluate(func, indexes):
    tps, tls, tas = [], [], []
    for batch_idx in indexes:
        predictions, lss, acc = func(batch_idx)

        tps.append(np.expand_dims(predictions, axis=0))
        tls.append(float(lss))
        tas.append(float(acc))
    predictions = np.concatenate(tps)
    lss = mean(tls)
    acc = mean(tas)

    return predictions, lss, acc


def evaluate_w2t(epoch, learning_rate, merged, model, sess, targets, writer):
    m = LogMessage()
    m.add('')
    m.add('Epoch: {epoch}'.format(epoch=epoch))
    m.add('  - learning rate   = {lr:e}'.format(lr=learning_rate.eval()))

    m.add('  Train data')

    def trn_eval(batch_idx):
        train_predictions, train_lss, train_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 1.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return train_predictions, train_lss, train_acc

    train_predictions, train_lss, train_acc = batch_evaluate(trn_eval, model.data.train_batch_indexes)

    m.add('    - accuracy      = {acc:f}'.format(acc=train_acc))
    m.add('    - loss          = {lss:f}'.format(lss=train_lss))

    m.add('  Dev data')

    def dev_eval(batch_idx):
        dev_predictions, dev_lss, dev_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 1.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return dev_predictions, dev_lss, dev_acc

    dev_predictions, dev_lss, dev_acc = batch_evaluate(dev_eval, model.data.dev_batch_indexes)

    m.add('    - accuracy      = {acc:f}'.format(acc=dev_acc))
    m.add('    - loss          = {lss:f}'.format(lss=dev_lss))

    m.add('  Test data')

    def tst_eval(batch_idx):
        test_predictions, test_lss, test_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 0.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return test_predictions, test_lss, test_acc

    test_predictions, test_lss, test_acc = batch_evaluate(tst_eval, model.data.test_batch_indexes)

    m.add('    - accuracy      = {acc:f}'.format(acc=test_acc))
    m.add('    - loss          = {lss:f}'.format(lss=test_lss))
    m.add()
    m.log()

    return train_predictions, train_acc, train_lss, \
           dev_predictions, dev_acc, dev_lss, \
           test_predictions, test_acc, test_lss


def evaluate_w2w(epoch, learning_rate, merged, model, sess, targets, use_inputs_prob, writer):
    m = LogMessage()
    m.add()
    m.add('Epoch: {epoch}'.format(epoch=epoch))
    m.add('  - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))
    m.add('  - use inputs prob = {uip:f}'.format(uip=use_inputs_prob))

    m.add('  Train data')

    def trn_eval(batch_idx):
        train_predictions, train_lss, train_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 1.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return train_predictions, train_lss, train_acc

    train_predictions, train_lss, train_acc = batch_evaluate(trn_eval, model.data.train_batch_indexes)

    m.add('    - use inputs prob = {uip:f}'.format(uip=1.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=train_acc))
    m.add('      - loss          = {lss:f}'.format(lss=train_lss))

    def trn_eval(batch_idx):
        train_predictions, test_lss, test_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 0.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return train_predictions, train_lss, train_acc

    train_predictions, train_lss, train_acc = batch_evaluate(trn_eval, model.data.train_batch_indexes)

    m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=train_acc))
    m.add('      - loss          = {lss:f}'.format(lss=train_lss))

    def dev_eval(batch_idx):
        dev_predictions, dev_lss, dev_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 0.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return dev_predictions, dev_lss, dev_acc

    dev_predictions, dev_lss, dev_acc = batch_evaluate(dev_eval, model.data.dev_batch_indexes)

    m.add('  Dev data')
    m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=dev_acc))
    m.add('      - loss          = {lss:f}'.format(lss=dev_lss))
    m.add()
    m.log()

    def tst_eval(batch_idx):
        test_predictions, test_lss, test_acc = sess.run(
            [model.predictions, model.loss, model.accuracy],
            feed_dict={
                model.batch_idx: batch_idx,
                model.use_inputs_prob: 0.0,
                model.dropout_keep_prob: 1.0,
                model.phase_train: False,
            }
        )
        return test_predictions, test_lss, test_acc

    test_predictions, test_lss, test_acc = batch_evaluate(tst_eval, model.data.test_batch_indexes)

    m.add('  Test data')
    m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=test_acc))
    m.add('      - loss          = {lss:f}'.format(lss=test_lss))
    m.add()
    m.log()

    return train_predictions, train_acc, train_lss, \
           dev_predictions, dev_acc, dev_lss, \
           test_predictions, test_acc, test_lss


def train(model, targets, idx2word_target):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          inter_op_parallelism_threads=FLAGS.threads,
                                          intra_op_parallelism_threads=FLAGS.threads,
                                          use_per_session_threads=True)) as sess:
        # Merge all the summaries and write them out to ./log
        merged_summaries = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(logging.exp_dir, sess.graph_def)
        saver = tf.train.Saver()

        # training
        t_vars = tf.trainable_variables()
        # t_vars = [v for v in t_vars if 'embedding_table' not in v.name] # all variables except embeddings
        learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)

        # train_op = tf.train.AdagradOptimizer(
        #         learning_rate=learning_rate,
        # )
        # train_op = AdamPlusCovOptimizer(
        train_op = AdamPlusOptimizer(
            learning_rate=learning_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            epsilon=FLAGS.epsilon,
            pow=FLAGS.pow,
            dense_regularization=FLAGS.dense_regularization,
            sparse_regularization=FLAGS.sparse_regularization,
            use_locking=False,
            name='trainer')

        learning_rate_decay_op = learning_rate.assign(learning_rate * FLAGS.decay)
        global_step = tf.Variable(0, trainable=False)
        gradients = tf.gradients(model.loss, t_vars)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        train_op = train_op.apply_gradients(zip(clipped_gradients, t_vars), global_step=global_step)

        tf.initialize_all_variables().run()

        # prepare batch indexes
        m = LogMessage()
        train_set_size = model.data.train_set_size
        m.add('Train set size: {d}'.format(d=train_set_size))
        batch_size = FLAGS.batch_size
        m.add('Batch size:     {d}'.format(d=batch_size))
        m.add('#Batches:       {d}'.format(d=len(model.data.train_batch_indexes)))
        m.log()

        train_accuracies, train_losses = [], []
        dev_accuracies, dev_losses = [], []
        test_accuracies, test_losses = [], []
        max_accuracy_epoch = 0
        use_inputs_prob = 1.0
        for epoch in range(FLAGS.max_epochs):
            # update the model
            LogMessage.write('Batch: ')
            for b, batch_idx in enumerate(model.data.train_batch_indexes):
                LogMessage.write(b)
                LogMessage.write(' ')
                sess.run(
                    [train_op],
                    feed_dict={
                        model.batch_idx: batch_idx,
                        model.use_inputs_prob: use_inputs_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        model.phase_train: True,
                    }
                )
            shuffle(model.data.train_batch_indexes)
            LogMessage.write('\n')

            # evaluate the model
            if FLAGS.task == 'w2t':
                train_predictions, train_acc, train_lss, \
                dev_predictions, dev_acc, dev_lss, \
                test_predictions, test_acc, test_lss = \
                    evaluate_w2t(epoch, learning_rate, merged_summaries, model, sess, targets, writer)
            else:
                train_predictions, train_acc, train_lss, \
                dev_predictions, dev_acc, dev_lss, \
                test_predictions, test_acc, test_lss = \
                    evaluate_w2w(epoch, learning_rate, merged_summaries, model, sess, targets, use_inputs_prob, writer)

            if epoch == 0 or dev_acc > max(dev_accuracies):
                max_accuracy_epoch = epoch

                model_fn = saver.save(sess, os.path.join(logging.exp_dir, "model.ckpt"))
                m = LogMessage()
                m.add('New max accuracy achieved on the dev data.')
                m.add("Model saved in file: {s}".format(s=model_fn))
                m.log()

                # save predictions on train, dev, and test sets
                if FLAGS.task == 'w2t':
                    predictions_argmax = np.argmax(train_predictions, 1)
                    log_predictions_w2t('predictions_train_set.txt', model,
                                        model.data.batch_histories, model.data.train_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)
                    predictions_argmax = np.argmax(dev_predictions, 1)
                    log_predictions_w2t('predictions_dev_set.txt', model,
                                        model.data.batch_histories, model.data.dev_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)
                    predictions_argmax = np.argmax(test_predictions, 1)
                    log_predictions_w2t('predictions_test_set.txt', model,
                                        model.data.batch_histories, model.data.test_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)
                else:
                    predictions_argmax = np.argmax(train_predictions, 2)
                    log_predictions_w2w('predictions_train_set.txt', model,
                                        model.data.batch_histories, model.data.train_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)
                    predictions_argmax = np.argmax(dev_predictions, 2)
                    log_predictions_w2w('predictions_dev_set.txt', model,
                                        model.data.batch_histories, model.data.dev_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)
                    predictions_argmax = np.argmax(test_predictions, 2)
                    log_predictions_w2w('predictions_test_set.txt', model,
                                        model.data.batch_histories, model.data.test_batch_indexes,
                                        predictions_argmax,
                                        targets, idx2word_target)

            m = LogMessage()
            m.add()
            m.add("Epoch with max accuracy on dev data: {d}".format(d=max_accuracy_epoch))
            m.add()
            m.log()

            # decrease learning rate if no improvement was seen over last 4 episodes.
            if len(train_losses) > 6 and train_lss >= max(train_losses[-4:]) + 1e-10:
                sess.run(learning_rate_decay_op)

            train_losses.append(train_lss)
            train_accuracies.append(train_acc)

            dev_losses.append(dev_lss)
            dev_accuracies.append(dev_acc)

            test_losses.append(test_lss)
            test_accuracies.append(test_acc)

            # stop when reached a threshold maximum or when no improvement of accuracy in the last 100 steps
            if train_acc > .999 or epoch > max_accuracy_epoch + 100:
                break

            use_inputs_prob *= FLAGS.use_inputs_prob_decay

            # save the results
            results = {
                'epoch': epoch,
                'max_accuracy_epoch_on_dev_data': max_accuracy_epoch,
                'train_loss': str(train_losses[max_accuracy_epoch]),
                'train_accuracy': str(train_accuracies[max_accuracy_epoch]),
                'dev_loss': str(dev_losses[max_accuracy_epoch]),
                'dev_accuracy': str(dev_accuracies[max_accuracy_epoch]),
                'test_loss': str(test_losses[max_accuracy_epoch]),
                'test_accuracy': str(test_accuracies[max_accuracy_epoch]),
            }

            LogExperiment(results)

    LogMessage(log_fn='.done', msg='done', time=True).log()


def main(run):
    start_experiment(run)

    if FLAGS.runs == 1:
        # set the seed to constant
        seed(0)
        tf.set_random_seed(1)

    graph = tf.Graph()

    with graph.as_default():
        with graph.device(device_for_node_gpu if FLAGS.gpu else device_for_node_cpu):
            if 'w2t' in FLAGS.model:
                FLAGS.task = 'w2t'
            if 'w2w' in FLAGS.model:
                FLAGS.task = 'w2w'

            m = LogMessage(time=True)
            m.add('-' * 120)
            m.add('End to End Neural Dialogue Manager')
            m.add('    runs                  = {runs}'.format(runs=FLAGS.runs))
            m.add('    threads               = {threads}'.format(threads=FLAGS.threads))
            m.add('    gpu                   = {gpu}'.format(gpu=FLAGS.gpu))
            m.add('    model                 = {model}'.format(model=FLAGS.model))
            m.add('    task                  = {t}'.format(t=FLAGS.task))
            m.add('    input                 = {i}'.format(i=FLAGS.input))
            m.add('    data_fraction         = {data_fraction}'.format(data_fraction=FLAGS.data_fraction))
            m.add('    train_data            = {train_data}'.format(train_data=FLAGS.train_data))
            m.add('    dev_data              = {dev_data}'.format(dev_data=FLAGS.dev_data))
            m.add('    test_data             = {test_data}'.format(test_data=FLAGS.test_data))
            m.add('    ontology              = {ontology}'.format(ontology=FLAGS.ontology))
            m.add('    database              = {database}'.format(database=FLAGS.database))
            m.add('    max_epochs            = {max_epochs}'.format(max_epochs=FLAGS.max_epochs))
            m.add('    batch_size            = {batch_size}'.format(batch_size=FLAGS.batch_size))
            m.add('    learning_rate         = {learning_rate:2e}'.format(learning_rate=FLAGS.learning_rate))
            m.add('    decay                 = {decay}'.format(decay=FLAGS.decay))
            m.add('    beta1                 = {beta1}'.format(beta1=FLAGS.beta1))
            m.add('    beta2                 = {beta2}'.format(beta2=FLAGS.beta2))
            m.add('    epsilon               = {epsilon}'.format(epsilon=FLAGS.epsilon))
            m.add('    pow                   = {pow}'.format(pow=FLAGS.pow))
            m.add('    dense_regularization  = {regularization}'.format(regularization=FLAGS.dense_regularization))
            m.add('    sparse_regularization = {regularization}'.format(regularization=FLAGS.sparse_regularization))
            m.add(
                '    max_gradient_norm     = {max_gradient_norm}'.format(max_gradient_norm=FLAGS.max_gradient_norm))
            m.add('    use_inputs_prob_decay = {use_inputs_prob_decay}'.format(
                use_inputs_prob_decay=FLAGS.use_inputs_prob_decay))
            m.add(
                '    dropout_keep_prob     = {dropout_keep_prob}'.format(dropout_keep_prob=FLAGS.dropout_keep_prob))
            m.add('-' * 120)
            m.log()

            data = dataset.DSTC2(
                input=FLAGS.input,
                data_fraction=FLAGS.data_fraction,
                train_data_fn=FLAGS.train_data,
                dev_data_fn=FLAGS.dev_data,
                test_data_fn=FLAGS.test_data,
                ontology_fn=FLAGS.ontology,
                database_fn=FLAGS.database,
                batch_size=FLAGS.batch_size
            )
            m = LogMessage()
            m.add('Database # rows:               {d}'.format(d=len(data.database)))
            m.add('Database # columns:            {d}'.format(d=len(data.database_word2idx.keys())))
            m.add('History vocabulary size:       {d}'.format(d=len(data.idx2word_history)))
            m.add('History args. vocabulary size: {d}'.format(d=len(data.idx2word_history_arguments)))
            m.add('State vocabulary size:         {d}'.format(d=len(data.idx2word_state)))
            m.add('Action vocabulary size:        {d}'.format(d=len(data.idx2word_action)))
            m.add('Action args. vocabulary size:  {d}'.format(d=len(data.idx2word_action_arguments)))
            m.add('Action tmpl. vocabulary size:  {d}'.format(d=len(data.idx2word_action_template)))
            m.add('-' * 120)
            m.log()

            if FLAGS.task == 'tracker':
                decoder_vocabulary_length = len(data.idx2word_state)
                idx2word_target = data.idx2word_state
                targets = data.batch_states
            elif FLAGS.task == 'w2w':
                decoder_vocabulary_length = len(data.idx2word_action)
                idx2word_target = data.idx2word_action
                targets = data.batch_actions
            elif FLAGS.task == 'w2t':
                decoder_vocabulary_length = len(data.idx2word_action_template)
                idx2word_target = data.idx2word_action_template
                targets = data.batch_actions_template
            else:
                raise Exception('Error: Unsupported task')

            if FLAGS.model == 'cnn-w2w':
                model = cnn_w2w.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn-w2w':
                model = rnn_w2w.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-w2t':
                model = cnn12_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-bn-w2t':
                model = cnn12_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-mp-bn-w2t':
                model = cnn12_mp_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn13-bn-w2t':
                model = cnn13_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn13-mp-bn-w2t':
                model = cnn13_mp_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-att-a-w2t':
                model = cnn12_att_a_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-bn-att-a-w2t':
                model = cnn12_bn_att_a_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-bn-att-a-bn-w2t':
                model = cnn12_bn_att_a_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-mp-bn-att-a-w2t':
                model = cnn12_mp_bn_att_a_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-att-b-w2t':
                model = cnn12_att_b_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn23-mp-bn-w2t':
                model = cnn23_mp_bn_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn1-w2t':
                model = rnn1_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn2-w2t':
                model = rnn2_w2t.Model(data, targets, decoder_vocabulary_length, FLAGS)
            else:
                raise Exception('Error: Unsupported model')

            train(model, targets, idx2word_target)


if __name__ == '__main__':
    flags.FLAGS._parse_flags()
    main = sys.modules['__main__'].main

    exp_dir = logging.prepare_experiment(FLAGS)

    ps = []
    for i in range(FLAGS.runs):
        print('Starting process {d}'.format(d=i))

        p = multiprocessing.Process(target=main, args=(i,))
        p.start()

        ps.append(p)

    summary_hash = 0
    while FLAGS.runs > 1:
        sleep(30)
        dev_loss, dev_accuracy = [], []
        epoch, max_accuracy_epoch_on_dev_data = [], []

        done_runs = 0
        for i, p in enumerate(ps):
            try:
                e = logging.read_experiment(i)
                epoch.append(int(e['epoch']))
                max_accuracy_epoch_on_dev_data.append(int(e['max_accuracy_epoch_on_dev_data']))
                dev_loss.append(float(e['dev_loss']))
                dev_accuracy.append(float(e['dev_accuracy']))
            except FileNotFoundError:
                pass

            if logging.experiment_done(i):
                # count number of finished runs
                done_runs += 1

        new_summary_hash = make_hash((epoch, max_accuracy_epoch_on_dev_data, dev_loss, dev_accuracy,))

        if len(epoch) and summary_hash != new_summary_hash:
            summary_hash = new_summary_hash

            # run only if we have some stats
            m = LogMessage(time=True)
            m.add('-' * 80)
            m.add('Experiment summary')
            m.add('  runs = {runs}'.format(runs=FLAGS.runs))
            m.add()
            m.add('  epoch min              = {d}'.format(d=min(epoch)))
            m.add('        max              = {d}'.format(d=max(epoch)))
            m.add('  max_accuracy_epoch min = {d}'.format(d=min(max_accuracy_epoch_on_dev_data)))
            m.add('                     max = {d}'.format(d=max(max_accuracy_epoch_on_dev_data)))
            m.add()
            m.add('  dev acc max            = {f:6f}'.format(f=max(dev_accuracy)))
            m.add('          mean           = {f:6f}'.format(f=mean(dev_accuracy)))
            if FLAGS.runs > 1:
                m.add('          stdev          = {f:6f}'.format(f=stdev(dev_accuracy)))
            m.add('          min            = {f:6f}'.format(f=min(dev_accuracy)))
            m.add()
            m.log()

        if done_runs >= len(ps):
            # stop this loop when all runs are finished
            break

    # for i, p in enumerate(ps):
    #     p.join()
    #     print('Joining process {d}'.format(d=i))

    print('All done')
