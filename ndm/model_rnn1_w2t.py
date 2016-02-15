#!/usr/bin/env python3

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell

from tfx.bricks import embedding, rnn, dense_to_one_hot, brnn, linear

from model import ModelW2T


class Model(ModelW2T):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        encoder_embedding_size = 16
        encoder_lstm_size = 16
        encoder_vocabulary_length = len(data.idx2word_history)
        encoder_sequence_length = data.train_set['histories'].shape[2]
        history_length = data.train_set['histories'].shape[1]

        action_templates_vocabulary_length = len(data.idx2word_action_template)

        with tf.name_scope('data'):
            batch_histories = tf.Variable(data.batch_histories, name='histories', trainable=False)
            batch_actions_template = tf.Variable(data.batch_actions_template, name='actions',
                                                 trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            actions_template = tf.gather(batch_actions_template, self.batch_idx)

        with tf.name_scope('model'):
            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(histories)[0]

            encoder_embedding = embedding(
                input=histories,
                length=encoder_vocabulary_length,
                size=encoder_embedding_size,
                name='encoder_embedding'
            )

            with tf.name_scope("UtterancesEncoder"):
                with tf.name_scope("RNNForwardUtteranceEncoderCell_1"):
                    cell_fwu_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=encoder_embedding_size,
                        use_peepholes=True
                    )
                    initial_state_fw_1 = cell_fwu_1.zero_state(batch_size, tf.float32)

                # the input data has this dimensions
                # [
                #   #batch,
                #   #utterance in a history (a dialogue),
                #   #word in an utterance (a sentence),
                #   embedding dimension
                # ]

                # encode all utterances along the word axis
                encoder_states_2d = []

                for utterance in range(history_length):
                    _, encoder_states = rnn(
                        cell=cell_fwu_1,
                        inputs=[encoder_embedding[:, utterance, word, :] for word in
                                range(encoder_sequence_length)],
                        initial_state=initial_state_fw_1,
                        name='RNNUtteranceForwardEncoder',
                        reuse=True if utterance > 0 else None
                    )

                    # print(encoder_states[-1])
                    encoder_states = tf.concat(1, tf.expand_dims(encoder_states[-1], 1))
                    # print(encoder_states)
                    encoder_states_2d.append(encoder_states)

                encoder_states_2d = tf.concat(1, encoder_states_2d)
                # print('encoder_states_2d', encoder_states_2d)

            with tf.name_scope("HistoryEncoder"):
                # encode all histories along the utterance axis
                with tf.name_scope("RNNForwardHistoryEncoderCell_1"):
                    cell_fwh_1 = LSTMCell(
                        num_units=encoder_lstm_size,
                        input_size=cell_fwu_1.state_size,
                        use_peepholes=True
                    )
                    initial_state_fw_1 = cell_fwh_1.zero_state(batch_size, tf.float32)

                _, encoder_states = rnn(
                    cell=cell_fwh_1,
                    inputs=[encoder_states_2d[:, utterance, :] for utterance in range(history_length)],
                    initial_state=initial_state_fw_1,
                    name='RNNHistoryForwardEncoder',
                    reuse=None
                )

            with tf.name_scope("Decoder"):
                linear_size = cell_fwh_1.state_size

                # decode all histories along the utterance axis
                activation = tf.nn.relu(encoder_states[-1])
                activation = tf.nn.dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=linear_size,
                    output_size=linear_size,
                    name='linear_projection_1'
                )
                activation = tf.nn.relu(projection)
                activation = tf.nn.dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=linear_size,
                    output_size=linear_size,
                    name='linear_projection_2'
                )
                activation = tf.nn.relu(projection)
                activation = tf.nn.dropout(activation, self.dropout_keep_prob)

                projection = linear(
                    input=activation,
                    input_size=linear_size,
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
            self.loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)),
                                       name='loss')
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(self.predictions, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', self.accuracy)
