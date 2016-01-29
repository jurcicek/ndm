#!/usr/bin/env python3

import math
import tensorflow as tf


def glorot_plus(n1, n2):
    return math.sqrt(6) / math.sqrt(float(n1 + n2))


def glorot_mul(n1, n2):
    return math.sqrt(3) / math.sqrt(float(n1 * n2))


def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    d = tf.nn.dropout(x, keep_prob, noise_shape, seed, name)

    d.size = getattr(x, 'size', None)

    return d

def pow_1(x, y):
    """Rises power of x as a complement of 1.

    :param x:
    :param y:
    :return:
    """
    return 1 - tf.pow(1-x, y)

def reduce_max(input, reduction_indices=None, keep_dims=False, name=None):
    r = tf.reduce_max(input, reduction_indices, keep_dims, name)

    r.size = getattr(input, 'size', None)

    return r


def linear(input, input_size, output_size, name='linear'):
    """Creates a linear transformation between two layers in a neural network.

    :param input: input into the linear block
    :param input_size: input dimension
    :param output_size: output dimension
    :param name: name of the operation
    """
    with tf.variable_scope(name):
        W = tf.get_variable(
                name='W',
                shape=[input_size, output_size],
                initializer=tf.random_uniform_initializer(-glorot_plus(input_size, output_size),
                                                          glorot_plus(input_size, output_size)),
        )
        b = tf.get_variable(
                name='B',
                shape=[output_size],
                initializer=tf.truncated_normal_initializer(stddev=1e-9 / glorot_mul(input_size, output_size)),
        )

        y = tf.matmul(input, W) + b

        y.input_size = input_size
        y.output_size = output_size
        y.W = W
        y.b = b

    return y


def embedding(input, length, size, name='embedding'):
    """Embedding transformation between discrete input and continuous vector representation.

    :param input: input 2-D tensor (e.g. rows are examples, columns are words)
    :param length: int, the lengths of the table representing the embeddings. This is equal to the size of the all discrete
        inputs.
    :param size: int, size of vector representing the discrete inputs.
    :param name: str, name of the operation
    """
    with tf.variable_scope(name):
        embedding_table = tf.get_variable(
                name='embedding_table',
                shape=[length, size],
                initializer=tf.random_uniform_initializer(-glorot_plus(length, size), glorot_plus(length, size)),
        )

        y = tf.gather(embedding_table, input)

        y.length = length
        y.size = size
        y.embedding_table = embedding_table

    return y


def multicolumn_embedding(columns, lengths, sizes, name='database_embedding'):
    """Embedding transformation between discrete input in multiple rows (database) and continuous vector representation.

    Note that each column has his own embedding table!

    :param database: input 2-D tensor (e.g. rows are entries in the database, columns are like database columns encoded
                     using a dictionary)
    :param lengths: int array, the lengths of the tables representing the embeddings. This is equal to the sizes of
                    the all discrete entries for individual columns.
    :param sizes: int array, sizes of vectors representing the discrete inputs.
    :param name: str, name of the operation
    """
    with tf.variable_scope(name):
        n_columns = len(lengths)

        columns = [columns[:, column] for column in range(n_columns)]

        columns_embeddings = []
        for i, column, length, size in zip(range(n_columns), columns, lengths, sizes):
            ce = embedding(
                    input=column,
                    length=length,
                    size=size,
                    name='database_embedding_column_{i}'.format(i=i)
            )

            columns_embeddings.append(ce)

        y = tf.concat(1, columns_embeddings)

        y.lengths = lengths
        y.sizes = sizes
        y.embedding_tables = columns_embeddings

    return y


def conv2d(input, filter, strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        W = tf.get_variable(
                name='W',
                shape=filter,
                # initializer=tf.truncated_normal_initializer(stddev=0.1)
                initializer=tf.truncated_normal_initializer() # this initialisation is much better!
        )

        y = tf.nn.relu(tf.nn.conv2d(input, W, strides=strides, padding='SAME'))

        y.filter = filter
        y.strides = strides
        y.size = filter[-1]
        y.W = W
        # y.b = b

    return y


def conv2d_bn(input, filter, phase_train, strides=[1, 1, 1, 1], name='conv2d_bn'):
    with tf.variable_scope(name):
        W = tf.get_variable(
                name='W',
                shape=filter,
                # initializer=tf.truncated_normal_initializer(stddev=0.1)
                initializer=tf.truncated_normal_initializer()
        )

        conv = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
        bn = batch_norm_conv(conv, filter[-1], phase_train)
        y = tf.nn.relu(bn)

        y.filter = filter
        y.strides = strides
        y.size = filter[-1]
        y.W = W

    return y


def max_pool(input, ksize, strides, name='max_pool'):
    with tf.name_scope(name):
        y = tf.nn.max_pool(input, ksize, strides, padding='SAME')
        y.ksize = ksize
        y.strides = strides

        y.size = getattr(input, 'size', None)
    return y


def batch_norm_conv(x, n_out, phase_train, name='bn', affine=True, decay=0.99, epsilon=1e-3):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        name:        string, variable scope
        affine:      whether to affine-transform outputs
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name):
        beta = tf.get_variable(
                name='beta',
                shape=[n_out],
                initializer=tf.constant_initializer(0.0)
        )
        gamma = tf.get_variable(
                name='gamma',
                shape=[n_out],
                initializer=tf.constant_initializer(1.0),
                trainable=affine
        )

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.python.control_flow_ops.cond(phase_train,
                                                    mean_var_with_update,
                                                    lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, epsilon, affine)
    return normed


def batch_norm_lin(x, n_out, phase_train, name='bn', affine=True, decay=0.9, epsilon=1e-3):
    """ A version of bach norm for a dense NN layer.

    :param x: 2D tensor
    :param n_out:
    :param phase_train:
    :param name:
    :param affine:
    :param decay:
    :param epsilon:
    :return:  batch-normalized output
    """
    x = tf.reshape(x, [-1, 1, 1, n_out])
    bn = batch_norm_conv(x, n_out, phase_train, name, affine, decay, epsilon)
    bn = tf.reshape(bn, [-1, n_out])

    return bn


def softmax_2d(input, n_classifiers, n_classes, name='softmax_2d'):
    with tf.name_scope(name):
        input = tf.reshape(input, [-1, n_classifiers, n_classes])
        e_x = tf.exp(input - tf.reduce_max(input, reduction_indices=2, keep_dims=True)) + 1e-10
        p_o_i = e_x / tf.reduce_sum(e_x, reduction_indices=2, keep_dims=True)

        return p_o_i


def rnn(cell, inputs, initial_state, name='RNN', reuse=False):
    """Forward recurrent neural network.

    :param cell: An instance of RNNCell
    :param inputs: A list of tensors, each a tensor of shape [batch_size, cell.input_size]
    :param initial_state: A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param name: A name of the variable scope for created or reused variables
    :param reuse: True if any created variables should be reused otherwise None.
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        outputs = []
        states = [initial_state]

        for j, input in enumerate(inputs):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                output, state = cell(input, states[-1])

                outputs.append(output)
                states.append(state)

    # remove the initial state
    states = states[1:]

    return outputs, states


def brnn(cell_fw, cell_bw, inputs, initial_state_fw, initial_state_bw=None, name='BidirectionalRNN', reuse=False):
    """Bidirectional recurrent neural network.

    :param cell_fw: An instance of RNNCell, for the forward pass
    :param cell_bw: An instance of RNNCell, for the backward pass
    :param inputs: A list of tensors, each a tensor of shape [batch_size, cell.input_size]
    :param initial_state_fw: A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param initial_state_bw: (Optional) A tensors, each a tensor of shape [batch_size, cell.state_size]
    :param name: A name of the variable scope for created or reused variables
    :param reuse: True if any created variables should be reused otherwise None.
    :return:
    """
    if not initial_state_bw:
        initial_state_bw = initial_state_fw

    with tf.variable_scope(name, reuse=reuse):
        outputs_fw, states_fw = rnn(cell_fw, inputs, initial_state_fw, name='ForwardRNN', reuse=reuse)
        outputs_bw, states_bw = rnn(cell_bw, reversed(inputs), initial_state_bw, name='BackwardRNN', reuse=reuse)

        # print(outputs_fw, states_fw)
        # print(outputs_bw, states_bw)

        outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(outputs_fw, reversed(outputs_bw))]
        states = [tf.concat(1, [fw, bw]) for fw, bw in zip(states_fw, reversed(states_bw))]

    return outputs, states


def rnn_decoder(cell, inputs, initial_state, embedding_size, embedding_length, sequence_length,
                name='RNNDecoder', reuse=False, use_inputs_prob=0.0, static_input=None):
    with tf.variable_scope(name, reuse=reuse):
        # print(tf.get_variable_scope().reuse, tf.get_variable_scope().name)
        with tf.name_scope("embedding"):
            batch_size = tf.shape(initial_state)[0]
            embedding_table = tf.get_variable(
                    name='embedding_table',
                    shape=[embedding_length, embedding_size],
                    initializer=tf.truncated_normal_initializer(stddev=glorot_mul(embedding_length, embedding_size)),
            )
            # 0 is index for _SOS_ (start of sentence symbol)
            initial_embedding = tf.gather(embedding_table, tf.zeros(tf.pack([batch_size]), tf.int32))

        states = [initial_state]
        outputs = []
        outputs_softmax = []
        decoder_outputs_argmax_embedding = []

        for j in range(sequence_length):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                # get input :
                #   either feedback the previous decoder argmax output
                #   or use the provided input (note that you have to use the previous input (index si therefore -1)
                input = initial_embedding
                if j > 0:
                    true_input = tf.gather(embedding_table, inputs[j - 1])
                    decoded_input = decoder_outputs_argmax_embedding[-1]
                    choice = tf.floor(tf.random_uniform([1], use_inputs_prob, 1 + use_inputs_prob, tf.float32))
                    input = choice * true_input + (1.0 - choice) * decoded_input

                if static_input:
                    input = tf.concat(1, [input, static_input])

                # print(tf.get_variable_scope().reuse, tf.get_variable_scope().name)
                output, state = cell(input, states[-1])

                projection = linear(
                        input=output,
                        input_size=cell.output_size,
                        output_size=embedding_length,
                        name='output_linear_projection'
                )

                outputs.append(projection)
                states.append(state)

                softmax = tf.nn.softmax(projection, name="output_softmax")
                # we do no compute the gradient trough argmax
                output_argmax = tf.stop_gradient(tf.argmax(softmax, 1))
                # we do no compute the gradient for embeddings when used with noisy argmax outputs

                output_argmax_embedding = tf.stop_gradient(tf.gather(embedding_table, output_argmax))
                decoder_outputs_argmax_embedding.append(output_argmax_embedding)

                outputs_softmax.append(tf.expand_dims(softmax, 1))

    # remove the initial state
    states = states[1:]

    return states, outputs, outputs_softmax


def dense_to_one_hot(labels, n_classes):
    with tf.name_scope('dense_to_one_hot'):
        indices = tf.where(tf.greater_equal(labels, tf.zeros_like(labels)))
        concated = tf.concat(1, [tf.to_int32(indices), tf.reshape(labels, [-1, 1])])
        dim = tf.concat(0, [tf.shape(labels), tf.reshape(n_classes, [-1])])

        one_hot_labels = tf.sparse_to_dense(
                concated,
                dim,
                1.0,
                0.0
        )

        return one_hot_labels


def device_for_node_cpu(n):
    if n.type == "MatMul":
        return "/cpu:0"
    else:
        return "/cpu:0"


def device_for_node_gpu_matmul(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"

# GPU matmul
# real    5m35.455s
# user    10m27.452s
# sys     2m50.512s

# GPU none
# real    4m46.137s
# user    8m37.684s
# sys     3m42.420s
