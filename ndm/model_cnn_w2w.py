#!/usr/bin/env python3
import sys

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell
from tfx.bricks import embedding, rnn_decoder, dense_to_one_hot, linear, conv2d, max_pool


class Model:
    def __init__(self, data, targets, decoder_vocabulary_length, FLAGS):
        dropout_keep_prob = tf.placeholder("float32", name='dropout_keep_prob')

        with tf.variable_scope("phase_train"):
            phase_train = tf.placeholder(tf.bool, name='phase_train')

        with tf.variable_scope("history_length"):
            history_length = data.train_set['histories'].shape[1]

        encoder_embedding_size = 32 * 4
        encoder_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("encoder_sequence_length"):
            encoder_sequence_length = data.train_set['histories'].shape[2]

        decoder_lstm_size = 16 * 2
        decoder_embedding_size = 16 * 2
        # decoder_vocabulary_length = len(data.idx2word_target)
        with tf.variable_scope("decoder_sequence_length"):
            decoder_sequence_length = targets.shape[2]

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
                    length=encoder_vocabulary_length,
                    size=encoder_embedding_size,
                    name='encoder_embedding'
            )

            with tf.name_scope("UtterancesEncoder"):
                conv3 = encoder_embedding
                # conv3 = conv2d(
                #         input=conv3,
                #         filter=[1, 3, encoder_embedding_size, encoder_embedding_size],
                #         name='conv_utt_size_3_layer_1'
                # )
                # conv_s3 = conv2d(
                #         input=conv_s3,
                #         filter=[1, 3, encoder_embedding_size, encoder_embedding_size],
                #         name='conv_utt_size_3_layer_2'
                # )
                # print(conv3)
                # k = encoder_sequence_length
                # mp_s3 = max_pool(conv_s3, ksize=[1, 1, k, 1], strides=[1, 1, k, 1])
                # print(mp_s3)

                # encoded_utterances = mp_s3
                encoded_utterances = tf.reduce_max(conv3, [2], keep_dims=True)

            with tf.name_scope("HistoryEncoder"):
                conv3 = encoded_utterances
                # conv3 = conv2d(
                #         input=conv3,
                #         filter=[3, 1, encoder_embedding_size, encoder_embedding_size],
                #         name='conv_hist_size_3_layer_1'
                # )
                # conv_s3 = conv2d(
                #         input=conv_s3,
                #         filter=[3, 1, encoder_embedding_size, encoder_embedding_size],
                #         name='conv_hist_size_3_layer_2'
                # )
                # print(conv3)
                # k = encoder_sequence_length
                # mp_s3 = max_pool(conv_s3, ksize=[1, 1, k, 1], strides=[1, 1, k, 1])
                # print(mp_s3)

                encoded_history = tf.reduce_max(conv3, [1, 2])

                # projection = linear(
                #         input=encoded_history,
                #         input_size=encoder_embedding_size,
                #         output_size=encoder_embedding_size,
                #         name='linear_projection_1'
                # )
                # encoded_history = tf.nn.relu(projection)
                # projection = linear(
                #         input=encoded_history,
                #         input_size=encoder_embedding_size,
                #         output_size=encoder_embedding_size,
                #         name='linear_projection_2'
                # )
                # encoded_history = tf.nn.relu(projection)
                # projection = linear(
                #         input=encoded_history,
                #         input_size=encoder_embedding_size,
                #         output_size=decoder_lstm_size * 2,
                #         name='linear_projection_3'
                # )
                # encoded_history = tf.nn.relu(projection)

            with tf.name_scope("Decoder"):
                use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')

                with tf.name_scope("RNNDecoderCell"):
                    cell = LSTMCell(
                            num_units=decoder_lstm_size,
                            input_size=decoder_embedding_size+encoder_embedding_size,
                            use_peepholes=True,
                    )
                    initial_state = cell.zero_state(batch_size, tf.float32)

                # decode all histories along the utterance axis
                final_encoder_state = encoded_history

                decoder_states, decoder_outputs, decoder_outputs_softmax = rnn_decoder(
                        cell=cell,
                        inputs=[targets[:, word] for word in range(decoder_sequence_length)],
                        static_input=final_encoder_state,
                        initial_state=initial_state, #final_encoder_state,
                        embedding_size=decoder_embedding_size,
                        embedding_length=decoder_vocabulary_length,
                        sequence_length=decoder_sequence_length,
                        name='RNNDecoder',
                        reuse=False,
                        use_inputs_prob=use_inputs_prob
                )

                predictions = tf.concat(1, decoder_outputs_softmax)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(targets, decoder_vocabulary_length)
            loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(predictions, 1e-10, 1.0)), name='loss')
            tf.scalar_summary('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(predictions, 2))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', accuracy)

        self.phase_train = phase_train

        self.data = data

        self.database = database

        self.batch_idx = batch_idx

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
