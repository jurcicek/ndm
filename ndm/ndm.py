#!/usr/bin/env python3
import os
import sys
from random import seed

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

import dataset
import model_cnn_w2w as cnn_w2w
import model_rnn_w2w as rnn_w2w
import model_cnn02_w2t as cnn02_w2t
import model_cnn12_w2t as cnn12_w2t
import model_cnn12_bn_w2t as cnn12_bn_w2t
import model_cnn12_mp_bn_w2t as cnn12_mp_bn_w2t
import model_cnn12_att_a_w2t as cnn12_att_a_w2t
import model_cnn12_bn_att_a_w2t as cnn12_bn_att_a_w2t
import model_cnn13_bn_w2t as cnn13_bn_w2t
import model_cnn13_mp_bn_w2t as cnn13_mp_bn_w2t
import model_cnn22_w2t as cnn2_w2t
import model_cnn23_mp_bn_w2t as cnn23_mp_bn_w2t
import model_rnn1_w2t as rnn1_w2t
import model_rnn2_w2t as rnn2_w2t

from tfx.bricks import device_for_node_cpu
from tfx.optimizers import AdamPlusOptimizer, AdamPlusCovOptimizer
from tfx.logging import start_experiment, LogMessage

import tfx.logging as logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'cnn-w2w', '"cnn-w2w" (convolutional network for state tracking - words 2 words ) | '
                                        '"rnn-w2w" (bidirectional recurrent network for state tracking - words 2 words) | '
                                        '"cnn02-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn12-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn12-bn_w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn12-mp-bn_w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn12-att-a-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                                        '"cnn12-bn-att-a-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                                        '"cnn13-bn-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn13-mp-bn-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn22-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn23-mp-bn-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"rnn1-w2t" (forward only recurrent network for state tracking - words 2 template | '
                                        '"rnn2-w2t" (bidirectional recurrent network for state tracking - words 2 template)')
flags.DEFINE_string('task', 'tracker', '"tracker" (dialogue state tracker) | '
                                       '"w2w" (word to word dialogue management) | '
                                       '"w2t" (word to template dialogue management)')
flags.DEFINE_string('input', 'asr', '"asr" automatically recognised user input | '
                                    '"trs" manually transcribed user input')
flags.DEFINE_string('train_data', './data.dstc2.train.json', 'The train data.')
flags.DEFINE_string('dev_data', './data.dstc2.dev.json', 'The development data.')
flags.DEFINE_string('test_data', './data.dstc2.test.json', 'The test data.')
flags.DEFINE_float('data_fraction', 0.1, 'The fraction of data to usd to train model.')
flags.DEFINE_string('ontology', './data.dstc2.ontology.json', 'The ontology defining slots and their values.')
flags.DEFINE_string('database', './data.dstc2.db.json', 'The backend database defining entries that can be queried.')
flags.DEFINE_integer('max_epochs', 100, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in a batch.')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
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


def log_predictions_w2t(log_fn, model, data_set, predictions_argmax, targets, idx2word_target):
    m = LogMessage(log_fn=log_fn)
    m.add('Shape of predictions: {s}'.format(s=predictions_argmax.shape))
    m.add('Argmax predictions')
    m.add()
    for history in range(0, predictions_argmax.shape[0]):
        m.add('History {d}'.format(d=history))

        for j in range(data_set['histories'].shape[1]):
            utterance = []
            for k in range(data_set['histories'].shape[2]):
                w = model.data.idx2word_history[data_set['histories'][history, j, k]]
                if w not in ['_SOS_', '_EOS_']:
                    utterance.append(w)
            if utterance:
                m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

        m.add('P  : {t:80}'.format(t=idx2word_target[predictions_argmax[history]]))
        m.add('T  : {t:80}'.format(t=idx2word_target[data_set[targets][history]]))
        m.add()
    m.log(print_console=False)


def log_predictions_w2w(log_fn, model, data_set, predictions_argmax, targets, idx2word_target):
    m = LogMessage(log_fn=log_fn)
    m.add('Shape of predictions: {s}'.format(s=predictions_argmax.shape))
    m.add('Argmax predictions')
    m.add()
    for history in range(0, predictions_argmax.shape[0]):
        m.add('History {d}'.format(d=history))

        for j in range(data_set['histories'].shape[1]):
            utterance = []
            for k in range(data_set['histories'].shape[2]):
                w = model.data.idx2word_history[data_set['histories'][history, j, k]]
                if w not in ['_SOS_', '_EOS_']:
                    utterance.append(w)
            if utterance:
                m.add('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

        prediction = []
        for j in range(predictions_argmax.shape[1]):
            w = idx2word_target[predictions_argmax[history, j]]
            if w not in ['_SOS_', '_EOS_']:
                prediction.append(w)

        m.add('P  : {t:80}'.format(t=' '.join(prediction)))

        target = []
        for j in range(data_set[targets].shape[1]):
            w = idx2word_target[data_set[targets][history, j]]
            if w not in ['_SOS_', '_EOS_']:
                target.append(w)

        m.add('T  : {t:80}'.format(t=' '.join(target)))
        m.add()
    m.log(print_console=False)


def evaluate_w2t(epoch, learning_rate, merged, model, sess, targets, writer):
    m = LogMessage()
    m.add('')
    m.add('Epoch: {epoch}'.format(epoch=epoch))
    m.add('  - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))
    m.log()

    m = LogMessage()
    m.add('  Train data')
    train_predictions, train_lss, train_acc = sess.run(
        [model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.train_set['histories'],
            model.histories_arguments: model.train_set['histories_arguments'],
            model.targets: model.train_set[targets],
            model.use_inputs_prob: 1.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    m.add('    - accuracy      = {acc:f}'.format(acc=train_acc))
    m.add('    - loss          = {lss:f}'.format(lss=train_lss))
    m.log()

    m = LogMessage()
    m.add('  Dev data')
    dev_predictions, dev_lss, dev_acc = sess.run(
        [model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.dev_set['histories'],
            model.histories_arguments: model.dev_set['histories_arguments'],
            model.targets: model.dev_set[targets],
            model.use_inputs_prob: 1.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    m.add('    - accuracy      = {acc:f}'.format(acc=dev_acc))
    m.add('    - loss          = {lss:f}'.format(lss=dev_lss))
    m.log()

    m = LogMessage()
    m.add('  Test data')
    summary, test_predictions, test_lss, test_acc = sess.run(
        [merged, model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.test_set['histories'],
            model.histories_arguments: model.test_set['histories_arguments'],
            model.targets: model.test_set[targets],
            model.use_inputs_prob: 0.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    writer.add_summary(summary, epoch)
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
    train_predictions, train_lss, train_acc = sess.run(
        [model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.train_set['histories'],
            model.histories_arguments: model.train_set['histories_arguments'],
            model.targets: model.train_set[targets],
            model.use_inputs_prob: 1.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    m.add('    - use inputs prob = {uip:f}'.format(uip=1.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=train_acc))
    m.add('      - loss          = {lss:f}'.format(lss=train_lss))
    train_predictions, test_lss, test_acc = sess.run(
        [model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.train_set['histories'],
            model.histories_arguments: model.train_set['histories_arguments'],
            model.targets: model.train_set[targets],
            model.use_inputs_prob: 0.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=test_acc))
    m.add('      - loss          = {lss:f}'.format(lss=test_lss))

    summary, dev_predictions, dev_lss, dev_acc = sess.run(
        [merged, model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.dev_set['histories'],
            model.histories_arguments: model.dev_set['histories_arguments'],
            model.targets: model.dev_set[targets],
            model.use_inputs_prob: 0.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    writer.add_summary(summary, epoch)
    m.add('  Dev data')
    m.add('    - use inputs prob = {uip:f}'.format(uip=0.0))
    m.add('      - accuracy      = {acc:f}'.format(acc=dev_acc))
    m.add('      - loss          = {lss:f}'.format(lss=dev_lss))
    m.add()
    m.log()

    test_predictions, test_lss, test_acc = sess.run(
        [model.predictions, model.loss, model.accuracy],
        feed_dict={
            model.database: model.data.database,
            model.histories: model.test_set['histories'],
            model.histories_arguments: model.test_set['histories_arguments'],
            model.targets: model.test_set[targets],
            model.use_inputs_prob: 0.0,
            model.dropout_keep_prob: 1.0,
            model.phase_train: False,
        }
    )
    writer.add_summary(summary, epoch)
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
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                          intra_op_parallelism_threads=2,
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
        train_set_size = model.train_set['histories'].shape[0]
        m.add('Train set size: {d}'.format(d=train_set_size))
        batch_size = FLAGS.batch_size
        m.add('Batch size:     {d}'.format(d=batch_size))
        m.add('#Batches:       {d}'.format(d=len(model.data.train_batch_indexes)))
        m.log()

        dev_previous_accuracies = []
        dev_previous_losses = []
        max_epoch = 0
        use_inputs_prob = 1.0
        for epoch in range(FLAGS.max_epochs):
            # update the model
            LogMessage.write('Batch: ')
            for b, batch in enumerate(model.data.iter_train_batches()):
                LogMessage.write(b)
                LogMessage.write(' ')
                sess.run(
                    [train_op],
                    feed_dict={
                        model.database: model.data.database,
                        model.histories: batch['histories'],
                        model.histories_arguments: batch['histories_arguments'],
                        model.targets: batch[targets],
                        model.use_inputs_prob: use_inputs_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        model.phase_train: True,
                    }
                )
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

            if epoch == 0 or dev_lss < min(dev_previous_losses):
                max_epoch = epoch

                model_fn = saver.save(sess, os.path.join(logging.exp_dir, "model.ckpt"))
                m = LogMessage()
                m.add('New max accuracy achieved on the dev data.')
                m.add("Model saved in file: {s}".format(s=model_fn))
                m.log()

                # save predictions on train, dev, and test sets
                if FLAGS.task == 'w2t':
                    predictions_argmax = np.argmax(train_predictions, 1)
                    log_predictions_w2t('predictions_train_set.txt', model, model.train_set, predictions_argmax,
                                        targets,
                                        idx2word_target)
                    predictions_argmax = np.argmax(dev_predictions, 1)
                    log_predictions_w2t('predictions_dev_set.txt', model, model.dev_set, predictions_argmax,
                                        targets,
                                        idx2word_target)
                    predictions_argmax = np.argmax(test_predictions, 1)
                    log_predictions_w2t('predictions_test_set.txt', model, model.test_set, predictions_argmax, targets,
                                        idx2word_target)
                else:
                    predictions_argmax = np.argmax(train_predictions, 2)
                    log_predictions_w2w('predictions_train_set.txt', model, model.train_set, predictions_argmax,
                                        targets,
                                        idx2word_target)
                    predictions_argmax = np.argmax(dev_predictions, 2)
                    log_predictions_w2w('predictions_dev_set.txt', model, model.dev_set, predictions_argmax,
                                        targets,
                                        idx2word_target)
                    predictions_argmax = np.argmax(test_predictions, 2)
                    log_predictions_w2w('predictions_test_set.txt', model, model.test_set, predictions_argmax, targets,
                                        idx2word_target)

            m = LogMessage()
            m.add()
            m.add("Epoch with min loss on dev data: {d}".format(d=max_epoch))
            m.add()
            m.log()

            # decrease learning rate if no improvement was seen over last 2 episodes.
            if len(dev_previous_losses) > 2 and dev_lss > max(dev_previous_losses[-2:]):
                sess.run(learning_rate_decay_op)
            dev_previous_losses.append(dev_lss)

            # stop when reached a threshold maximum or when no improvement on loss in the last 100 steps
            if dev_acc > 0.9999 or \
                len(dev_previous_losses) > 120 and min(dev_previous_losses[:-100]) < min(dev_previous_losses[-100:]):
                break

            dev_previous_accuracies.append(dev_acc)

        use_inputs_prob *= FLAGS.use_inputs_prob_decay


def main(_):
    start_experiment(FLAGS)

    tf.set_random_seed(1)
    graph = tf.Graph()

    with graph.as_default():
        with graph.device(device_for_node_cpu):
            m = LogMessage(time=True)
            m.add('-' * 120)
            m.add('End to End Neural Dialogue Manager')
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
            m.add('    learning_rate         = {learning_rate}'.format(learning_rate=FLAGS.learning_rate))
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
                targets = 'states'
            elif FLAGS.task == 'w2w':
                decoder_vocabulary_length = len(data.idx2word_action)
                idx2word_target = data.idx2word_action
                targets = 'actions'
            elif FLAGS.task == 'w2t':
                decoder_vocabulary_length = len(data.idx2word_action_template)
                idx2word_target = data.idx2word_action_template
                targets = 'actions_template'
            else:
                raise Exception('Error: Unsupported task')

            if FLAGS.model == 'cnn-w2w':
                model = cnn_w2w.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn-w2w':
                model = rnn_w2w.Model(data, targets, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn02-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn02-w2t only supports ONLY tasks w2t!')
                model = cnn02_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-w2t only supports ONLY tasks w2t!')
                model = cnn12_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-bn-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-bn-w2t only supports ONLY tasks w2t!')
                model = cnn12_bn_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-mp-bn-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-mp-bn-w2t only supports ONLY tasks w2t!')
                model = cnn12_mp_bn_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn13-bn-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn13-bn-w2t only supports ONLY tasks w2t!')
                model = cnn13_bn_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn13-mp-bn-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn13-mp-bn-w2t only supports ONLY tasks w2t!')
                model = cnn13_mp_bn_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-att-a-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-att-a-w2t only supports ONLY tasks w2t!')
                model = cnn12_att_a_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-bn-att-a-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-bn-att-a-w2t only supports ONLY tasks w2t!')
                model = cnn12_bn_att_a_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn12-att-b-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn12-att-b-w2t only supports ONLY tasks w2t!')
                model = cnn12_att_b_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn22-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn22-w2t only supports ONLY tasks w2t!')
                model = cnn2_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn23-mp-bn-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn23-mp-bn-w2t only supports ONLY tasks w2t!')
                model = cnn23_mp_bn_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn1-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model rnn1-w2t only supports ONLY tasks w2t!')
                model = rnn1_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'rnn2-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model rnn2-w2t only supports ONLY tasks w2t!')
                model = rnn2_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            else:
                raise Exception('Error: Unsupported model')

            train(model, targets, idx2word_target)


if __name__ == '__main__':
    seed(0)
    tf.app.run()
