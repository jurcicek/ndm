#!/usr/bin/env python3

import tensorflow as tf

from tfx.bricks import embedding, dense_to_one_hot, linear, dropout, reduce_max, batch_norm_lin, conv2d_bn, \
    pow_1, softmax_2d

from model import ModelW2TArgs


class Model(ModelW2TArgs):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        conv_mul = 2
        histories_embedding_size = 16
        histories_vocabulary_length = len(data.idx2word_history)
        history_length = data.train_set['histories'].shape[1]

        action_templates_vocabulary_length = len(data.idx2word_action_template)
        action_templates_embedding_size = 8

        num_actions_arguments = data.batch_actions_arguments.shape[2]
        actions_arguments_vocabulary_length = len(data.idx2word_action_arguments)

        with tf.name_scope('data'):
            batch_histories = tf.Variable(data.batch_histories, name='histories',
                                          trainable=False)
            batch_actions_template = tf.Variable(data.batch_actions_template, name='actions',
                                                 trainable=False)
            batch_action_arguments = tf.Variable(data.batch_actions_arguments, name='actions_arguments',
                                                 trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            actions_template = tf.gather(batch_actions_template, self.batch_idx)
            actions_arguments = tf.gather(batch_action_arguments, self.batch_idx)

        with tf.name_scope('model'):
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

                # first encode decoded action template and teh true action template
                choice = tf.floor(tf.random_uniform([1], self.use_inputs_prob, 1 + self.use_inputs_prob, tf.float32))

                prediction_action_argmax = tf.stop_gradient(tf.argmax(self.predictions_action, 1))
                predicted_action_templates_embedding = embedding(
                    input=prediction_action_argmax,
                    length=action_templates_vocabulary_length,
                    size=action_templates_embedding_size,
                    name='action_templates_embedding'
                )

                true_action_template_embedding = tf.gather(predicted_action_templates_embedding.embedding_table, actions_template)
                predicted_action_templates_embedding = tf.stop_gradient(predicted_action_templates_embedding)

                action_templates_embedding = choice * true_action_template_embedding + (1.0 - choice) * predicted_action_templates_embedding

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
