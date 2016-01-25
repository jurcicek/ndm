#!/usr/bin/env python3
import sys

import tensorflow as tf

from tf_ext.bricks import embedding, dense_to_one_hot, linear, conv2d, max_pool, multicolumn_embedding


class Model:
    def __init__(self, data, decoder_vocabulary_length, FLAGS):
        with tf.variable_scope("history_length"):
            history_length = data.train_set['histories'].shape[1]

        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("histories_utterance_length"):
            histories_utterance_length = data.train_set['histories'].shape[2]

        histories_arguments_embedding_size = 8
        histories_arguments_vocabulary_length = len(data.idx2word_history_arguments)
        with tf.variable_scope("histories_arguments_length"):
            histories_arguments_length = data.train_set['histories_arguments'].shape[1]

        # inference model
        with tf.name_scope('model'):
            database = tf.placeholder("int32", name='database')
            histories = tf.placeholder("int32", name='histories')
            histories_arguments = tf.placeholder("int32", name='histories_arguments')
            targets = tf.placeholder("int32", name='true_targets')
            use_dropout_prob = tf.placeholder("float32", name='use_dropout_prob')

            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(histories)[0]

            database_embedding = multicolumn_embedding(
                    columns=database,
                    lengths=[len(i2w) for i2w in [data.database_idx2word[column] for column in data.database_columns]],
                    sizes=[8 for column in data.database_columns],  # all columns have the same size
                    name='database_embedding'
            )

            histories_embedding = embedding(
                    input=histories,
                    length=histories_vocabulary_length,
                    size=histories_embedding_size,
                    name='histories_embedding'
            )

            histories_arguments_embedding = embedding(
                    input=histories_arguments,
                    length=histories_arguments_vocabulary_length,
                    size=histories_arguments_embedding_size,
                    name='histories_arguments_embedding'
            )

            with tf.name_scope("UtterancesEncoder"):
                conv3 = histories_embedding
                conv3 = conv2d(
                        input=conv3,
                        filter=[1, 3, histories_embedding_size, histories_embedding_size],
                        name='conv_utt_size_3_layer_1'
                )
                conv3 = conv2d(
                        input=conv3,
                        filter=[1, 3, histories_embedding_size, histories_embedding_size],
                        name='conv_utt_size_3_layer_2'
                )
                # print(conv3)
                # k = encoder_sequence_length
                # mp = max_pool(conv3, ksize=[1, 1, k, 1], strides=[1, 1, k, 1])
                # print(mp)
                # encoded_utterances = mp

                encoded_utterances = tf.reduce_max(conv3, [2], keep_dims=True)

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, histories_embedding_size, histories_embedding_size],
                        name='conv_hist_size_3_layer_1'
                )
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, histories_embedding_size, histories_embedding_size],
                        name='conv_hist_size_3_layer_2'
                )
                # print(conv3)
                # k = encoder_sequence_length
                # mp = max_pool(conv3, ksize=[1, 1, k, 1], strides=[1, 1, k, 1])
                # print(mp)
                # encoded_history = tf.reshape(mp, [-1, encoder_embedding_size])

                encoded_history = tf.reduce_max(conv3, [1, 2])

            with tf.name_scope("DatabaseAttention"):
                histories_arguments_embedding = tf.reshape(histories_arguments_embedding, [-1, 10])
                database_embedding = tf.reshape(database_embedding, [-1, 10])

                attention = tf.concat(1, [encoded_history, histories_arguments_embedding, database_embedding])
                attention = tf.reshape(attention, [batch_size, -1])


            with tf.name_scope("Decoder"):
                use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')

                # decode all histories along the utterance axis
                activation = tf.nn.relu(encoded_history)

                projection = linear(
                        input=activation,
                        input_size=histories_embedding_size,
                        output_size=histories_embedding_size,
                        name='linear_projection_1'
                )
                activation = tf.nn.relu(projection)

                projection = linear(
                        input=activation,
                        input_size=histories_embedding_size,
                        output_size=histories_embedding_size,
                        name='linear_projection_2'
                )
                activation = tf.nn.relu(projection)

                projection = linear(
                        input=activation,
                        input_size=histories_embedding_size,
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
        self.encoder_sequence_length = histories_utterance_length
        self.histories = histories
        self.histories_arguments = histories_arguments
        self.targets = targets
        self.use_dropout_prob = use_dropout_prob
        self.batch_size = batch_size
        self.use_inputs_prob = use_inputs_prob
        self.predictions = predictions
        self.loss = loss
        self.accuracy = accuracy
