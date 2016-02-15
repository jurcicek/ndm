#!/usr/bin/env python3
import sys

import tensorflow as tf

from model import ModelW2T
from tfx.bricks import embedding, dense_to_one_hot, linear, conv2d, dropout, reduce_max, batch_norm_lin, conv2d_bn, \
    pow_1, max_pool


class Model(ModelW2T):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        conv_mul = 2
        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        histories_utterance_length = data.train_set['histories'].shape[2]
        history_length = data.train_set['histories'].shape[1]

        action_templates_vocabulary_length = len(data.idx2word_action_template)

        with tf.name_scope('data'):
            batch_histories = tf.Variable(data.batch_histories, name='histories',
                                          trainable=False)
            batch_actions_template = tf.Variable(data.batch_actions_template, name='actions',
                                                 trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            actions_template = tf.gather(batch_actions_template, self.batch_idx)

        with tf.name_scope('model'):
            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(histories)[0]

            encoder_embedding = embedding(
                    input=histories,
                    length=histories_vocabulary_length,
                    size=histories_embedding_size,
                    name='encoder_embedding'
            )

            with tf.name_scope("UtterancesEncoder"):
                conv3 = encoder_embedding
                # conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[1, 3, conv3.size, conv3.size * conv_mul],
                        phase_train=self.phase_train,
                        name='conv_utt_size_3_layer_1'
                )

                encoded_utterances = reduce_max(conv3, [2], keep_dims=True)

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        phase_train=self.phase_train,
                        name='conv_hist_size_3_layer_1'
                )
                conv3 = max_pool(conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1])
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        phase_train=self.phase_train,
                        name='conv_hist_size_3_layer_2'
                )

                encoded_history = reduce_max(conv3, [1, 2])

            with tf.name_scope("Decoder"):
                second_to_last_user_utterance = encoded_utterances[:, history_length - 3, 0, :]
                last_system_utterance = encoded_utterances[:, history_length - 2, 0, :]
                last_user_utterance = encoded_utterances[:, history_length - 1, 0, :]

                dialogue_state = tf.concat(
                        1,
                        [
                            encoded_history,
                            last_user_utterance,
                            last_system_utterance,
                            second_to_last_user_utterance,
                        ],
                        name='dialogue_state'
                )
                dialogue_state_size = conv3.size + \
                                      3 * histories_embedding_size * conv_mul

                activation = tf.nn.relu(dialogue_state)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_1'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, self.phase_train, name='linear_projection_1_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_2'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, self.phase_train, name='linear_projection_2_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=action_templates_vocabulary_length,
                        name='linear_projection_3'
                )
                self.predictions = tf.nn.softmax(projection, name="softmax_output")
                # print(self.predictions)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(actions_template, action_templates_vocabulary_length)
            self.loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)), name='loss')
            # self.loss = tf.reduce_mean(- one_hot_labels * tf.log(self.predictions), name='loss')
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(self.predictions, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', self.accuracy)
