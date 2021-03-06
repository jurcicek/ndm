#!/usr/bin/env python3
import sys

import tensorflow as tf

from model import ModelW2T
from tfx.bricks import embedding, dense_to_one_hot, linear, conv2d, multicolumn_embedding, \
    glorot_mul, reduce_max, dropout, pow_1


class Model(ModelW2T):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        database_column_embedding_size = 8
        n_database_columns = len(data.database_columns)

        conv_mul = 2
        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        histories_utterance_length = data.train_set['histories'].shape[2]
        history_length = data.train_set['histories'].shape[1]

        histories_arguments_embedding_size = 8
        histories_arguments_vocabulary_length = len(data.idx2word_history_arguments)
        n_histories_arguments = data.train_set['histories_arguments'].shape[1]

        action_templates_vocabulary_length = len(data.idx2word_action_template)

        with tf.name_scope('data'):
            database = tf.Variable(data.database, name='database',
                                   trainable=False)

            batch_histories = tf.Variable(data.batch_histories, name='histories',
                                          trainable=False)
            batch_histories_arguments = tf.Variable(data.batch_histories_arguments, name='histories_arguments',
                                                    trainable=False)
            batch_actions_template = tf.Variable(data.batch_actions_template, name='actions',
                                                 trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            histories_arguments = tf.gather(batch_histories_arguments, self.batch_idx)
            actions_template = tf.gather(batch_actions_template, self.batch_idx)

        with tf.name_scope('model'):
            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(histories)[0]

            database_embedding = multicolumn_embedding(
                    columns=database,
                    lengths=[len(i2w) for i2w in [data.database_idx2word[column] for column in data.database_columns]],
                    sizes=[database_column_embedding_size for column in data.database_columns],
                    # all columns have the same size
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
                # conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d(
                        input=conv3,
                        filter=[1, 3, conv3.size, conv3.size * conv_mul],
                        name='conv_utt_size_3_layer_1'
                )

                encoded_utterances = reduce_max(conv3, [2], keep_dims=True, name='encoded_utterances')

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        name='conv_hist_size_3_layer_1'
                )
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d(
                        input=conv3,
                        filter=[3, 1, conv3.size, conv3.size * conv_mul],
                        name='conv_hist_size_3_layer_2'
                )

                encoded_history = reduce_max(conv3, [1, 2], name='encoded_history')
                # print(encoded_history)

            with tf.name_scope("DatabaseAttention"):
                histories_arguments_embedding = tf.reshape(
                        histories_arguments_embedding,
                        [-1, n_histories_arguments * histories_arguments_embedding_size],
                        name='histories_arguments_embedding'
                )
                # print(histories_arguments_embedding)

                history_predicate = tf.concat(
                        1,
                        [encoded_history, histories_arguments_embedding],
                        name='history_predicate'
                )
                print(history_predicate)

                att_W_nx = conv3.size + n_histories_arguments * histories_arguments_embedding_size
                att_W_ny = n_database_columns * database_column_embedding_size

                att_W = tf.get_variable(
                        name='attention_W',
                        shape=[att_W_nx, att_W_ny],
                        initializer=tf.random_uniform_initializer(
                                -glorot_mul(att_W_nx, att_W_ny),
                                glorot_mul(att_W_nx, att_W_ny)
                        ),
                )
                hp_x_att_W = tf.matmul(history_predicate, att_W)
                attention_scores = tf.matmul(hp_x_att_W, database_embedding, transpose_b=True)
                attention = tf.nn.softmax(attention_scores, name="attention_softmax")
                print(attention)

                attention_max = tf.reduce_max(attention, reduction_indices=1, keep_dims=True)
                attention_min = tf.reduce_min(attention, reduction_indices=1, keep_dims=True)
                attention_mean = tf.reduce_mean(attention_scores, reduction_indices=1, keep_dims=True)
                attention_feat = tf.concat(1, [attention_max, attention_mean, attention_min], name='attention_feat')
                attention_feat_size = 3
                print(attention_feat)

                db_result = tf.matmul(attention, database_embedding, name='db_result')
                db_result_size = att_W_ny
                print(db_result)

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
                            attention_feat,
                            db_result
                        ],
                        name='dialogue_state'
                )
                dialogue_state_size = conv3.size + \
                                      3 * histories_embedding_size * conv_mul + \
                                      attention_feat_size + \
                                      db_result_size

                activation = tf.nn.relu(dialogue_state)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_1'
                )
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=dialogue_state_size,
                        name='linear_projection_2'
                )
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                        input=activation,
                        input_size=dialogue_state_size,
                        output_size=action_templates_vocabulary_length,
                        name='linear_projection_3'
                )
                self.predictions = tf.nn.softmax(projection, name="predictions")
                # print(self.predictions)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(actions_template, action_templates_vocabulary_length)
            self.loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)), name='loss')
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(self.predictions, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', self.accuracy)
