#!/usr/bin/env python3
import sys

import tensorflow as tf

from model import ModelW2T, ModelW2TArgs
from tfx.bricks import embedding, dense_to_one_hot, linear, conv2d, multicolumn_embedding, \
    glorot_mul, reduce_max, dropout, conv2d_bn, batch_norm_lin, pow_1, softmax_2d


class Model(ModelW2TArgs):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        database_column_embedding_size = 8
        n_database_columns = len(data.database_columns)

        conv_mul = 2
        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        history_length = data.train_set['histories'].shape[1]

        histories_arguments_embedding_size = 8
        histories_arguments_vocabulary_length = len(data.idx2word_history_arguments)
        n_histories_arguments = data.train_set['histories_arguments'].shape[1]

        action_templates_vocabulary_length = len(data.idx2word_action_template)
        action_templates_embedding_size = 8

        num_actions_arguments = data.batch_actions_arguments.shape[2]
        actions_arguments_vocabulary_length = len(data.idx2word_action_arguments)

        with tf.name_scope('data'):
            database = tf.Variable(data.database, name='database',
                                   trainable=False)

            batch_histories = tf.Variable(data.batch_histories, name='histories',
                                          trainable=False)
            batch_histories_arguments = tf.Variable(data.batch_histories_arguments, name='histories_arguments',
                                                    trainable=False)
            batch_actions_template = tf.Variable(data.batch_actions_template, name='actions',
                                                 trainable=False)
            batch_action_arguments = tf.Variable(data.batch_actions_arguments, name='actions_arguments',
                                                 trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            histories_arguments = tf.gather(batch_histories_arguments, self.batch_idx)
            actions_template = tf.gather(batch_actions_template, self.batch_idx)
            actions_arguments = tf.gather(batch_action_arguments, self.batch_idx)

        with tf.name_scope('model'):
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
                conv3 = conv2d_bn(
                    input=conv3,
                    filter=[1, 3, conv3.size, conv3.size * conv_mul],
                    phase_train=self.phase_train,
                    name='conv_utt_size_3_layer_1'
                )

                encoded_utterances = reduce_max(conv3, [2], keep_dims=True, name='encoded_utterances')

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                    input=conv3,
                    filter=[3, 1, conv3.size, conv3.size * conv_mul],
                    phase_train=self.phase_train,
                    name='conv_hist_size_3_layer_1'
                )
                conv3 = dropout(conv3, pow_1(self.dropout_keep_prob, 2))
                conv3 = conv2d_bn(
                    input=conv3,
                    filter=[3, 1, conv3.size, conv3.size * conv_mul],
                    phase_train=self.phase_train,
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
                # print(history_predicate)

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
                # print(attention_feat)

                db_result = tf.matmul(attention, database_embedding, name='db_result')
                db_result_size = att_W_ny
                # print(db_result)

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
                dialogue_state_size = (
                    conv3.size +
                    3 * histories_embedding_size * conv_mul +
                    attention_feat_size +
                    db_result_size +
                    0
                )

                dialogue_state = tf.nn.relu(dialogue_state)
                dialogue_state = dropout(dialogue_state, self.dropout_keep_prob)

                # action prediction
                projection = linear(
                    input=dialogue_state,
                    input_size=dialogue_state_size,
                    output_size=dialogue_state_size,
                    name='linear_projection_1'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, self.phase_train,
                                            name='linear_projection_1_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=dialogue_state_size,
                    output_size=dialogue_state_size,
                    name='linear_projection_2'
                )
                projection = batch_norm_lin(projection, dialogue_state_size, self.phase_train,
                                            name='linear_projection_2_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=dialogue_state_size,
                    output_size=action_templates_vocabulary_length,
                    name='linear_projection_3_predictions_action'
                )
                self.predictions_action = tf.nn.softmax(projection, name="softmax_output_prediction_action")

                # argument prediction

                # first encode decoded action template
                prediction_action_argmax = tf.argmax(self.predictions_action, 1)
                action_templates_embedding = embedding(
                    input=prediction_action_argmax,
                    length=action_templates_vocabulary_length,
                    size=action_templates_embedding_size,
                    name='action_templates_embedding'
                )

                dialogue_state_action_template = tf.concat(
                    1,
                    [
                        dialogue_state,
                        action_templates_embedding
                    ],
                    name='dialogue_state_action_template'
                )
                dialogue_state_action_template_size = (
                    dialogue_state_size +
                    action_templates_embedding_size
                )

                # condition on the dialogue state and the decoded template
                projection = linear(
                    input=dialogue_state_action_template,
                    input_size=dialogue_state_action_template_size,
                    output_size=dialogue_state_action_template_size,
                    name='linear_projection_1_predictions_arguments'
                )
                projection = batch_norm_lin(projection, dialogue_state_action_template_size, self.phase_train,
                                            name='linear_projection_1_predictions_arguments_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=dialogue_state_action_template_size,
                    output_size=dialogue_state_action_template_size,
                    name='linear_projection_2_predictions_arguments'
                )
                projection = batch_norm_lin(projection, dialogue_state_action_template_size, self.phase_train,
                                            name='linear_projection_2_predictions_arguments_bn')
                activation = tf.nn.relu(projection)
                activation = dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=dialogue_state_action_template_size,
                    output_size=num_actions_arguments * actions_arguments_vocabulary_length,
                    name='linear_projection_3_predictions_arguments'
                )
                self.predictions_arguments = softmax_2d(
                    input=projection,
                    n_classifiers=num_actions_arguments,
                    n_classes=actions_arguments_vocabulary_length,
                    name="softmax_2d_predictions_arguments")

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels_action = dense_to_one_hot(actions_template, action_templates_vocabulary_length)
            one_hot_labels_arguments = dense_to_one_hot(actions_arguments, actions_arguments_vocabulary_length)

            loss_action = tf.reduce_mean(
                - one_hot_labels_action * tf.log(tf.clip_by_value(self.predictions_action, 1e-10, 1.0)),
                name='loss'
            )
            loss_arguments = tf.reduce_mean(
                - one_hot_labels_arguments * tf.log(tf.clip_by_value(self.predictions_arguments, 1e-10, 1.0)),
                name='loss'
            )

            self.loss = loss_action + loss_arguments

            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction_action = tf.equal(
                tf.argmax(one_hot_labels_action, 1),
                tf.argmax(self.predictions_action, 1)
            )
            self.accuracy_action = tf.reduce_mean(tf.cast(correct_prediction_action, 'float'))
            tf.scalar_summary('accuracy_action', self.accuracy_action)

            correct_prediction_arguments = tf.equal(tf.argmax(one_hot_labels_arguments, 2),
                                                    tf.argmax(self.predictions_arguments, 2))
            self.accuracy_arguments = tf.reduce_mean(tf.cast(correct_prediction_arguments, 'float'))
            tf.scalar_summary('accuracy_arguments', self.accuracy_arguments)
