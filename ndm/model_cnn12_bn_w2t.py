#!/usr/bin/env python3
import sys

import tensorflow as tf

from tfx.bricks import embedding, dense_to_one_hot, linear, conv2d, dropout, reduce_max, batch_norm_lin, conv2d_bn, \
    pow_1


class Model:
    def __init__(self, data, targets, decoder_vocabulary_length, FLAGS):
        dropout_keep_prob = tf.placeholder("float32", name='dropout_keep_prob')

        with tf.variable_scope("phase_train"):
            phase_train = tf.placeholder(tf.bool, name='phase_train')

        with tf.variable_scope("history_length"):
            history_length = data.train_set['histories'].shape[1]

        conv_mul = 2
        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("encoder_sequence_length"):
            histories_utterance_length = data.train_set['histories'].shape[2]

        with tf.name_scope('data'):
            batch_idx = tf.placeholder("int32", name='batch_idx')

            database = tf.Variable(data.database, name='database', trainable=False)

            batch_histories = tf.Variable(data.batch_histories, name='histories', trainable=False)
            batch_histories_arguments = tf.Variable(data.batch_histories_arguments, name='histories_arguments', trainable=False)
            batch_targets = tf.Variable(targets, name='targets', trainable=False)

            histories = tf.gather(batch_histories, batch_idx)
            histories_arguments = tf.gather(batch_histories_arguments, batch_idx)
            targets = tf.gather(batch_targets, batch_idx)

        # inference model
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
                # conv3 = dropout(conv3, pow_1(dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[1, 3, conv3.size, conv3.size * conv_mul],
                        phase_train=phase_train,
                        name='conv_utt_size_3_layer_1'
                )

                encoded_utterances = reduce_max(conv3, [2], keep_dims=True)

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = dropout(conv3, pow_1(dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        phase_train=phase_train,
                        name='conv_hist_size_3_layer_1'
                )
                conv3 = dropout(conv3, pow_1(dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        phase_train=phase_train,
                        name='conv_hist_size_3_layer_2'
                )

                encoded_history = reduce_max(conv3, [1, 2])

            with tf.name_scope("Decoder"):
                use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')

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
                activation = dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_1'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, phase_train, name='linear_projection_1_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_2'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, phase_train, name='linear_projection_2_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=decoder_vocabulary_length,
                        name='linear_projection_3'
                )
                predictions = tf.nn.softmax(projection, name="softmax_output")
                # print(predictions)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(targets, decoder_vocabulary_length)
            loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(predictions, 1e-10, 1.0)), name='loss')
            # loss = tf.reduce_mean(- one_hot_labels * tf.log(predictions), name='loss')
            tf.scalar_summary('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(predictions, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', accuracy)

        self.dropout_keep_prob = dropout_keep_prob
        self.phase_train = phase_train

        self.data = data

        self.database = database

        self.batch_idx = batch_idx

        self.history_length = history_length
        self.encoder_sequence_length = histories_utterance_length
        self.histories = histories
        self.histories_arguments = histories_arguments
        self.attention = None  # attention
        self.db_result = None  # db_result
        self.targets = targets
        self.batch_size = batch_size
        self.use_inputs_prob = use_inputs_prob
        self.predictions = predictions
        self.loss = loss
        self.accuracy = accuracy
