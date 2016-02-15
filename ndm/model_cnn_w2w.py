#!/usr/bin/env python3

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMCell

from tfx.bricks import embedding, rnn_decoder, dense_to_one_hot, linear, conv2d, max_pool

from model import ModelW2W


class Model(ModelW2W):
    def __init__(self, data, FLAGS):
        super(Model, self).__init__(data, FLAGS)

        encoder_embedding_size = 32 * 4
        encoder_vocabulary_length = len(data.idx2word_history)

        decoder_lstm_size = 16 * 2
        decoder_embedding_size = 16 * 2
        decoder_sequence_length = data.batch_actions.shape[2]
        decoder_vocabulary_length = len(data.idx2word_action)

        with tf.name_scope('data'):
            batch_histories = tf.Variable(data.batch_histories, name='histories', trainable=False)
            batch_actions = tf.Variable(data.batch_actions, name='actions', trainable=False)

            histories = tf.gather(batch_histories, self.batch_idx)
            actions = tf.gather(batch_actions, self.batch_idx)

        with tf.name_scope('model'):
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
                with tf.name_scope("RNNDecoderCell"):
                    cell = LSTMCell(
                        num_units=decoder_lstm_size,
                        input_size=decoder_embedding_size + encoder_embedding_size,
                        use_peepholes=True,
                    )
                    initial_state = cell.zero_state(batch_size, tf.float32)

                # decode all histories along the utterance axis
                final_encoder_state = encoded_history

                decoder_states, decoder_outputs, decoder_outputs_softmax = rnn_decoder(
                    cell=cell,
                    inputs=[actions[:, word] for word in range(decoder_sequence_length)],
                    static_input=final_encoder_state,
                    initial_state=initial_state,  # final_encoder_state,
                    embedding_size=decoder_embedding_size,
                    embedding_length=decoder_vocabulary_length,
                    sequence_length=decoder_sequence_length,
                    name='RNNDecoder',
                    reuse=False,
                    use_inputs_prob=self.use_inputs_prob
                )

                self.predictions = tf.concat(1, decoder_outputs_softmax)

        if FLAGS.print_variables:
            for v in tf.trainable_variables():
                print(v.name)

        with tf.name_scope('loss'):
            one_hot_labels = dense_to_one_hot(actions, decoder_vocabulary_length)
            self.loss = tf.reduce_mean(- one_hot_labels * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)),
                                       name='loss')
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(one_hot_labels, 2), tf.argmax(self.predictions, 2))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
            tf.scalar_summary('accuracy', self.accuracy)
