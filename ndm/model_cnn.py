#!/usr/bin/env python3
import sys

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell
from tf_ext.bricks import embedding, rnn_decoder, dense_to_one_hot, linear, conv2d, max_pool


class CNN:
    def __init__(self, data, FLAGS):
        with tf.variable_scope("history_length"):
            history_length = data.train_set['features'].shape[1]

        encoder_embedding_size = 32 * 4
        encoder_vocabulary_length = len(data.idx2word_history)
        with tf.variable_scope("encoder_sequence_length"):
            encoder_sequence_length = data.train_set['features'].shape[2]

        decoder_lstm_size = 16 * 2
        decoder_embedding_size = 16 * 2
        decoder_vocabulary_length = len(data.idx2word_target)
        with tf.variable_scope("decoder_sequence_length"):
            decoder_sequence_length = data.train_set['targets'].shape[1]

        # inference model
        with tf.name_scope('model'):
            features = tf.placeholder("int32", name='features')
            targets = tf.placeholder("int32", name='true_targets')
            use_dropout_prob = tf.placeholder("float32", name='use_dropout_prob')

            with tf.variable_scope("batch_size"):
                batch_size = tf.shape(features)[0]

            encoder_embedding = embedding(
                    input=features,
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

                targets_given_features = tf.concat(1, decoder_outputs_softmax)
                # print(p_o_i)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(targets, decoder_vocabulary_length)
            loss = tf.reduce_mean(- one_hot_labels * tf.log(targets_given_features), name='loss')
            for v in tf.trainable_variables():
                for n in ['/W_', '/W:', '/B:']:
                    if n in v.name:
                        print('Regularization using', v.name)
                        loss += FLAGS.regularization * tf.reduce_mean(tf.pow(v, 2))
            tf.scalar_summary('loss', loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(targets_given_features, 2))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', accuracy)

        self.data = data
        self.train_set = data.train_set
        self.test_set = data.test_set
        self.idx2word_history = data.idx2word_history
        self.word2idx_history = data.word2idx_history
        self.idx2word_target = data.idx2word_target
        self.word2idx_target = data.word2idx_target

        self.history_length = history_length
        self.encoder_sequence_length = encoder_sequence_length
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.use_inputs_prob = use_inputs_prob
        self.targets_given_features = targets_given_features
        self.loss = loss
        self.accuracy = accuracy
