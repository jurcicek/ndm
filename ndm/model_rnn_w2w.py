#!/usr/bin/env python3
import sys

sys.path.extend(['..'])

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell
from tf_ext.bricks import embedding, rnn, rnn_decoder, dense_to_one_hot, brnn


class Model:
    def __init__(self, data, targets, decoder_vocabulary_length, FLAGS):
        with tf.variable_scope("history_length"):
            history_length = data.train_set['histories'].shape[1]

        encoder_lstm_size = 16
        encoder_embedding_size = 16 * 2
        encoder_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("encoder_sequence_length"):
            encoder_sequence_length = data.train_set['histories'].shape[2]

        decoder_lstm_size = 16
        decoder_embedding_size = 16
        # decoder_vocabulary_length = len(data.idx2word_target)
        with tf.variable_scope("decoder_sequence_length"):
            decoder_sequence_length = data.train_set[targets].shape[1]

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
                with tf.name_scope("RNNForwardUtteranceEncoderCell_1"):
                    cell_fw_1 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=encoder_embedding_size,
                            use_peepholes=True
                    )
                    initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

                with tf.name_scope("RNNBackwardUtteranceEncoderCell_1"):
                    cell_bw_1 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=encoder_embedding_size,
                            use_peepholes=True
                    )
                    initial_state_bw_1 = cell_bw_1.zero_state(batch_size, tf.float32)

                with tf.name_scope("RNNForwardUtteranceEncoderCell_2"):
                    cell_fw_2 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                            use_peepholes=True
                    )
                    initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

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
                    encoder_outputs, _ = brnn(
                            cell_fw=cell_fw_1,
                            cell_bw=cell_bw_1,
                            inputs=[encoder_embedding[:, utterance, word, :] for word in
                                    range(encoder_sequence_length)],
                            initial_state_fw=initial_state_fw_1,
                            initial_state_bw=initial_state_bw_1,
                            name='RNNUtteranceBidirectionalLayer',
                            reuse=True if utterance > 0 else None
                    )

                    _, encoder_states = rnn(
                            cell=cell_fw_2,
                            inputs=encoder_outputs,
                            initial_state=initial_state_fw_2,
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
                    cell_fw_1 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=cell_fw_2.state_size,
                            use_peepholes=True
                    )
                    initial_state_fw_1 = cell_fw_1.zero_state(batch_size, tf.float32)

                with tf.name_scope("RNNBackwardHistoryEncoderCell_1"):
                    cell_bw_1 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=cell_fw_2.state_size,
                            use_peepholes=True
                    )
                    initial_state_bw_1 = cell_fw_2.zero_state(batch_size, tf.float32)

                with tf.name_scope("RNNForwardHistoryEncoderCell_2"):
                    cell_fw_2 = LSTMCell(
                            num_units=encoder_lstm_size,
                            input_size=cell_fw_1.output_size + cell_bw_1.output_size,
                            use_peepholes=True
                    )
                    initial_state_fw_2 = cell_fw_2.zero_state(batch_size, tf.float32)

                encoder_outputs, _ = brnn(
                        cell_fw=cell_fw_1,
                        cell_bw=cell_bw_1,
                        inputs=[encoder_states_2d[:, utterance, :] for utterance in range(history_length)],
                        initial_state_fw=initial_state_fw_1,
                        initial_state_bw=initial_state_bw_1,
                        name='RNNHistoryBidirectionalLayer',
                        reuse=None
                )

                _, encoder_states = rnn(
                        cell=cell_fw_2,
                        inputs=encoder_outputs,
                        initial_state=initial_state_fw_2,
                        name='RNNHistoryForwardEncoder',
                        reuse=None
                )

            with tf.name_scope("Decoder"):
                use_inputs_prob = tf.placeholder("float32", name='use_inputs_prob')

                with tf.name_scope("RNNDecoderCell"):
                    cell = LSTMCell(
                            num_units=decoder_lstm_size,
                            input_size=decoder_embedding_size+cell_fw_2.state_size,
                            use_peepholes=True,
                    )
                    initial_state = cell.zero_state(batch_size, tf.float32)

                # decode all histories along the utterance axis
                final_encoder_state = encoder_states[-1]

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
                # print(p_o_i)

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
