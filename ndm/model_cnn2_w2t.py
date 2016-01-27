#!/usr/bin/env python3
import sys

import tensorflow as tf

from tf_ext.bricks import embedding, dense_to_one_hot, linear, conv2d, max_pool


class Model:
    def __init__(self, data, decoder_vocabulary_length, FLAGS):
        with tf.variable_scope("history_length"):
            history_length = data.train_set['histories'].shape[1]

        encoder_embedding_size = 16
        encoder_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("encoder_sequence_length"):
            encoder_sequence_length = data.train_set['histories'].shape[2]

        # inference model
        with tf.name_scope('model'):
            database = tf.placeholder("int32", name='database')
            histories = tf.placeholder("int32", name='histories')
            histories_arguments = tf.placeholder("int32", name='histories_arguments')
            targets = tf.placeholder("int32", name='true_targets')
            dropout_keep_prob = tf.placeholder("float32", name='dropout_keep_prob')

            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(histories)[0]

            encoder_embedding = embedding(
                    input=histories,
                    length=encoder_vocabulary_length,
                    size=encoder_embedding_size,
                    name='encoder_embedding'
            )

            with tf.name_scope("UtterancesEncoder"):
                conv3 = encoder_embedding
                conv3 = tf.nn.dropout(conv3, dropout_keep_prob)
                conv3 = conv2d(
                        input=conv3,
                        filter=[1, 3, encoder_embedding_size, encoder_embedding_size],
                        name='conv_utt_size_3_layer_1'
                )
                conv3 = tf.nn.dropout(conv3, dropout_keep_prob)
                conv3 = conv2d(
                        input=conv3,
                        filter=[1, 3, encoder_embedding_size, encoder_embedding_size],
                        name='conv_utt_size_3_layer_2'
                )

                encoded_utterances = tf.reduce_max(conv3, [2], keep_dims=True)

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = tf.nn.dropout(conv3, dropout_keep_prob)
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, encoder_embedding_size, encoder_embedding_size],
                        name='conv_hist_size_3_layer_1'
                )
                conv3 = tf.nn.dropout(conv3, dropout_keep_prob)
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, encoder_embedding_size, encoder_embedding_size],
                        name='conv_hist_size_3_layer_2'
                )

                encoded_history = tf.reduce_max(conv3, [1, 2])

            with tf.name_scope("Decoder"):
                use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')

                # decode all histories along the utterance axis
                activation = tf.nn.relu(encoded_history)
                activation = tf.nn.dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=encoder_embedding_size,
                        output_size=encoder_embedding_size,
                        name='linear_projection_1'
                )
                activation = tf.nn.relu(projection)
                activation = tf.nn.dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=encoder_embedding_size,
                        output_size=encoder_embedding_size,
                        name='linear_projection_2'
                )
                activation = tf.nn.relu(projection)
                activation = tf.nn.dropout(activation, dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=encoder_embedding_size,
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

        self.data = data

        self.database = database

        self.train_set = data.train_set
        self.dev_set = data.dev_set
        self.test_set = data.test_set

        self.history_length = history_length
        self.encoder_sequence_length = encoder_sequence_length
        self.histories = histories
        self.histories_arguments = histories_arguments
        self.attention = None #attention
        self.db_result = None #db_result
        self.targets = targets
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.use_inputs_prob = use_inputs_prob
        self.predictions = predictions
        self.loss = loss
        self.accuracy = accuracy
