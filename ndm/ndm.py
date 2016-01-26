#!/usr/bin/env python3
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf

import dataset
import model_cnn_w2w as cnn_w2w
import model_rnn_w2w as rnn_w2w
import model_cnn0_w2t as cnn0_w2t
import model_cnn1_w2t as cnn1_w2t
import model_cnn2_w2t as cnn2_w2t
import model_cnn2_att_w2t as cnn2_att_w2t
import model_rnn1_w2t as rnn1_w2t
import model_rnn2_w2t as rnn2_w2t

from tf_ext.bricks import device_for_node_cpu
from tf_ext.optimizers import AdamPlusOptimizer, AdamPlusCovOptimizer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'cnn-w2w', '"cnn-w2w" (convolutional network for state tracking - words 2 words ) | '
                                        '"rnn-w2w" (bidirectional recurrent network for state tracking - words 2 words) | '
                                        '"cnn0-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn1-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn2-w2t" (convolutional network for state tracking - words 2 template | '
                                        '"cnn2-att-w2t" (convolutional network for state tracking with attention model - words 2 template | '
                                        '"rnn1-w2t" (forward only recurrent network for state tracking - words 2 template | '
                                        '"rnn2-w2t" (bidirectional recurrent network for state tracking - words 2 template)')
flags.DEFINE_string('task', 'tracker', '"tracker" (dialogue state tracker) | '
                                       '"w2w" (word to word dialogue management) | '
                                       '"w2t" (word to template dialogue management)')
flags.DEFINE_string('input', 'asr', '"asr" automatically recognised user input | '
                                    '"trs" manually transcribed user input')
flags.DEFINE_string('train_data', './data.dstc2.traindev.json', 'The train data the model should be trained on.')
flags.DEFINE_string('test_data', './data.dstc2.test.json', 'The test data the model should be trained on.')
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
flags.DEFINE_float('pow', 0.999, 'AdamPlusOptimizer pow.')
flags.DEFINE_float('dense_regularization', 1e-16, 'Weight of regularization for dense updates.')
flags.DEFINE_float('sparse_regularization', 1e-16, 'Weight of regularization foir sparse updates.')
flags.DEFINE_float('max_gradient_norm', 5e0, 'Clip gradients to this norm.')
flags.DEFINE_float('use_inputs_prob_decay', 0.999, 'Decay of the probability of using '
                                                   'the true targets during generation.')
flags.DEFINE_boolean('print_variables', False, 'Print all trainable variables.')

"""
This code shows how to build and train a neural dialogue manager.
There are several models available:
 1) bidirectional RNN for encoding utterances
 2) convolutional neural network for encoding utterances
"""


def train(model, targets, idx2word_target):
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                          intra_op_parallelism_threads=2,
                                          use_per_session_threads=True)) as sess:
        # Merge all the summaries and write them out to ./log
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./log', sess.graph_def)
        saver = tf.train.Saver()

        # training
        t_vars = tf.trainable_variables()
        # t_vars = [v for v in t_vars if 'embedding_table' not in v.name] # all variables except embeddings
        learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False)

        train_op = AdamPlusOptimizer(
        # train_op = AdamPlusCovOptimizer(
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
        train_set_size = model.train_set['histories'].shape[0]
        print('Train set size:', train_set_size)
        batch_size = FLAGS.batch_size
        print('Batch size:', batch_size)
        print('#Batches:', len(model.data.train_batch_indexes))
        # print('Batch indexes', batch_indexes)

        previous_accuracies = []
        previous_losses = []
        use_inputs_prob = 1.0
        for epoch in range(FLAGS.max_epochs):
            print('Batch: ', end=' ', flush=True)
            for b, batch in enumerate(model.data.iter_train_batches()):
                print(b, end=' ', flush=True)
                sess.run(
                        [train_op],
                        feed_dict={
                            model.database: model.data.database,
                            model.histories: batch['histories'],
                            model.histories_arguments: batch['histories_arguments'],
                            model.targets: batch[targets],
                            model.use_inputs_prob: use_inputs_prob,
                        }
                )
            print()

            if epoch % max(min(int(FLAGS.max_epochs / 100), 100), 1) == 0:
                if FLAGS.task == 'w2t':
                    print()
                    print('Epoch: {epoch}'.format(epoch=epoch))
                    print('  - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))
                    print('  Train data')
                    lss, acc = sess.run([model.loss, model.accuracy],
                                        feed_dict={
                                            model.database: model.data.database,
                                            model.histories: model.train_set['histories'],
                                            model.histories_arguments: model.train_set['histories_arguments'],
                                            model.targets: model.train_set[targets],
                                            model.use_inputs_prob: 1.0,
                                        })
                    print('    - accuracy      = {acc:f}'.format(acc=acc))
                    print('    - loss          = {lss:f}'.format(lss=lss))
                    summary, lss, acc = sess.run([merged, model.loss, model.accuracy],
                                                 feed_dict={
                                                     model.database: model.data.database,
                                                     model.histories: model.test_set['histories'],
                                                     model.histories_arguments: model.test_set['histories_arguments'],
                                                     model.targets: model.test_set[targets],
                                                     model.use_inputs_prob: 0.0,
                                                 })
                    writer.add_summary(summary, epoch)
                    print('  Test data')
                    print('    - accuracy      = {acc:f}'.format(acc=acc))
                    print('    - loss          = {lss:f}'.format(lss=lss))
                    print()

                    # decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and lss > max(previous_losses[-3:]):
                        sess.run(learning_rate_decay_op)
                    previous_losses.append(lss)

                    # stop when reached a threshold maximum or when no improvement in the last 20 steps
                    previous_accuracies.append(acc)
                    if acc > 0.9999 or max(previous_accuracies) > max(previous_accuracies[-20:]):
                        break
                else:
                    print()
                    print('Epoch: {epoch}'.format(epoch=epoch))
                    print('  - learning rate   = {lr:f}'.format(lr=learning_rate.eval()))
                    print('  - use inputs prob = {uip:f}'.format(uip=use_inputs_prob))
                    print('  Train data')
                    lss, acc = sess.run([model.loss, model.accuracy],
                                        feed_dict={
                                            model.database: model.data.database,
                                            model.histories: model.train_set['histories'],
                                            model.histories_arguments: model.train_set['histories_arguments'],
                                            model.targets: model.train_set[targets],
                                            model.use_inputs_prob: 1.0,
                                        })
                    print('    - use inputs prob = {uip:f}'.format(uip=1.0))
                    print('      - accuracy      = {acc:f}'.format(acc=acc))
                    print('      - loss          = {lss:f}'.format(lss=lss))
                    lss, acc = sess.run([model.loss, model.accuracy],
                                        feed_dict={
                                            model.database: model.data.database,
                                            model.histories: model.train_set['histories'],
                                            model.histories_arguments: model.train_set['histories_arguments'],
                                            model.targets: model.train_set[targets],
                                            model.use_inputs_prob: 0.0,
                                        })
                    print('    - use inputs prob = {uip:f}'.format(uip=0.0))
                    print('      - accuracy      = {acc:f}'.format(acc=acc))
                    print('      - loss          = {lss:f}'.format(lss=lss))
                    summary, lss, acc = sess.run([merged, model.loss, model.accuracy],
                                                 feed_dict={
                                                     model.database: model.data.database,
                                                     model.histories: model.test_set['histories'],
                                                     model.histories_arguments: model.test_set['histories_arguments'],
                                                     model.targets: model.test_set[targets],
                                                     model.use_inputs_prob: 0.0,
                                                 })
                    writer.add_summary(summary, epoch)
                    print('  Test data')
                    print('    - use inputs prob = {uip:f}'.format(uip=0.0))
                    print('      - accuracy      = {acc:f}'.format(acc=acc))
                    print('      - loss          = {lss:f}'.format(lss=lss))
                    print()

                    # decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and lss > max(previous_losses[-3:]):
                        sess.run(learning_rate_decay_op)
                    previous_losses.append(lss)

                    # stop when reached a threshold maximum or when no improvement in the last 20 steps
                    previous_accuracies.append(acc)
                    if acc > 0.9999 or max(previous_accuracies) > max(previous_accuracies[-20:]):
                        break

            use_inputs_prob *= FLAGS.use_inputs_prob_decay

        save_path = saver.save(sess, ".cnn-model.ckpt")
        print()
        print("Model saved in file: %s" % save_path)
        print()

        # print('Test features')
        # print(test_set['histories'])
        # print('Test targets')
        print('Shape of targets:', model.test_set[targets].shape)
        # print(test_set[targets])
        print('Predictions')
        predictions = sess.run(model.predictions,
                               feed_dict={
                                   model.database: model.data.database,
                                   model.histories: model.test_set['histories'],
                                   model.histories_arguments: model.test_set['histories_arguments'],
                                   model.targets: model.test_set[targets],
                                   model.use_inputs_prob: 0.0,
                               })

        if FLAGS.task == 'w2w':
            predictions_argmax = np.argmax(predictions, 2)
            print('Shape of predictions:', predictions.shape)
            print('Argmax predictions')
            print()
            for history in range(0, predictions_argmax.shape[0],
                                 max(int(predictions_argmax.shape[0] * 0.05), 1)):
                print('History', history)

                for j in range(model.test_set['histories'].shape[1]):
                    utterance = []
                    for k in range(model.test_set['histories'].shape[2]):
                        w = model.data.idx2word_history[model.test_set['histories'][history, j, k]]
                        if w not in ['_SOS_', '_EOS_']:
                            utterance.append(w)
                    if utterance:
                        print('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

                prediction = []
                for j in range(predictions_argmax.shape[1]):
                    w = idx2word_target[predictions_argmax[history, j]]
                    if w not in ['_SOS_', '_EOS_']:
                        prediction.append(w)

                print('P  : {t:80}'.format(t=' '.join(prediction)))

                target = []
                for j in range(model.test_set[targets].shape[1]):
                    w = idx2word_target[model.test_set[targets][history, j]]
                    if w not in ['_SOS_', '_EOS_']:
                        target.append(w)

                print('T  : {t:80}'.format(t=' '.join(target)))
                print()
        else:
            predictions_argmax = np.argmax(predictions, 1)
            print('Shape of predictions:', predictions.shape)
            print('Argmax predictions')
            print()
            for history in range(0, predictions_argmax.shape[0],
                                 max(int(predictions_argmax.shape[0] * 0.05), 1)):
                print('History', history)

                for j in range(model.test_set['histories'].shape[1]):
                    utterance = []
                    for k in range(model.test_set['histories'].shape[2]):
                        w = model.data.idx2word_history[model.test_set['histories'][history, j, k]]
                        if w not in ['_SOS_', '_EOS_']:
                            utterance.append(w)
                    if utterance:
                        print('U {j}: {c:80}'.format(j=j, c=' '.join(utterance)))

                print('P  : {t:80}'.format(t=idx2word_target[predictions_argmax[history]]))
                print('T  : {t:80}'.format(t=idx2word_target[model.test_set[targets][history]]))
                print()


def main(_):
    graph = tf.Graph()

    with graph.as_default():
        with graph.device(device_for_node_cpu):
            print('-' * 120)
            print('End to End Neural Dialogue Manager')
            print('    model                 = {model}'.format(model=FLAGS.model))
            print('    task                  = {t}'.format(t=FLAGS.task))
            print('    input                 = {i}'.format(i=FLAGS.input))
            print('    train_data            = {train_data}'.format(train_data=FLAGS.train_data))
            print('    data_fraction         = {data_fraction}'.format(data_fraction=FLAGS.data_fraction))
            print('    test_data             = {test_data}'.format(test_data=FLAGS.test_data))
            print('    ontology              = {ontology}'.format(ontology=FLAGS.ontology))
            print('    database              = {database}'.format(database=FLAGS.database))
            print('    max_epochs            = {max_epochs}'.format(max_epochs=FLAGS.max_epochs))
            print('    batch_size            = {batch_size}'.format(batch_size=FLAGS.batch_size))
            print('    learning_rate         = {learning_rate}'.format(learning_rate=FLAGS.learning_rate))
            print('    decay                 = {decay}'.format(decay=FLAGS.decay))
            print('    beta1                 = {beta1}'.format(beta1=FLAGS.beta1))
            print('    beta2                 = {beta2}'.format(beta2=FLAGS.beta2))
            print('    epsilon               = {epsilon}'.format(epsilon=FLAGS.epsilon))
            print('    pow                   = {pow}'.format(pow=FLAGS.pow))
            print('    dense_regularization  = {regularization}'.format(regularization=FLAGS.dense_regularization))
            print('    sparse_regularization = {regularization}'.format(regularization=FLAGS.sparse_regularization))
            print('    max_gradient_norm     = {max_gradient_norm}'.format(max_gradient_norm=FLAGS.max_gradient_norm))
            print('    use_inputs_prob_decay = {use_inputs_prob_decay}'.format(
                    use_inputs_prob_decay=FLAGS.use_inputs_prob_decay))
            print('-' * 120)
            data = dataset.DSTC2(
                    mode=FLAGS.task,
                    input=FLAGS.input,
                    train_data_fn=FLAGS.train_data,
                    data_fraction=FLAGS.data_fraction,
                    test_data_fn=FLAGS.test_data,
                    ontology_fn=FLAGS.ontology,
                    database_fn=FLAGS.database,
                    batch_size=FLAGS.batch_size
            )

            print('Database # rows:              ', len(data.database))
            print('Database # columns:           ', len(data.database_word2idx.keys()))
            print('History vocabulary size:      ', len(data.idx2word_history))
            print('History args. vocabulary size:', len(data.idx2word_history_arguments))
            print('State vocabulary size:        ', len(data.idx2word_state))
            print('Action vocabulary size:       ', len(data.idx2word_action))
            print('Action args. vocabulary size: ', len(data.idx2word_action_arguments))
            print('Action tmpl. vocabulary size: ', len(data.idx2word_action_template))
            print('-' * 120)

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
            elif FLAGS.model == 'cnn0-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn1-w2t only supports ONLY tasks w2t!')
                model = cnn0_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn1-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn1-w2t only supports ONLY tasks w2t!')
                model = cnn1_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn2-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn2-w2t only supports ONLY tasks w2t!')
                model = cnn2_w2t.Model(data, decoder_vocabulary_length, FLAGS)
            elif FLAGS.model == 'cnn2-att-w2t':
                if FLAGS.task != 'w2t':
                    raise Exception('Error: Model cnn2-att-w2t only supports ONLY tasks w2t!')
                model = cnn2_att_w2t.Model(data, decoder_vocabulary_length, FLAGS)
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
    tf.app.run()
