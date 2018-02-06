import tensorflow as tf


def lstm_cell(lstm_out_dimension, name):
    """Define LSTM cell
    """
    with tf.variable_scope(name):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_out_dimension)


def lstm(lstm_out_dimension, layers, embeddings, max_length, batch_size, name, is_training):
    """Build model
    """
    with tf.name_scope(name):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [ lstm_cell(lstm_out_dimension, 'lstm_cell_' + str(d)) for d in range(layers) ],
            state_is_tuple=True)
        input_placeholder = tf.placeholder(tf.int32,
                                           [batch_size, max_length], name='X_' + name)
        embedded_inputs = tf.nn.embedding_lookup(embeddings, input_placeholder)
        seqlen = tf.placeholder(tf.int32, [batch_size], name='seqlen_' + name)
        init_state = stacked_lstm.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm,
                                                     embedded_inputs,
                                                     initial_state=init_state,
                                                     dtype=tf.float32,
                                                     sequence_length=seqlen,
                                                     scope='question_embedding_rnn')

        # Add dropout, as the model otherwise quickly overfits.
        keep_prob = tf.constant(0.7)
        rnn_outputs = tf.cond(is_training,
                              lambda: tf.nn.dropout(rnn_outputs, keep_prob),
                              lambda: rnn_outputs)
        for idx, weights in enumerate(stacked_lstm.weights):
            tf.summary.histogram('weights' + str(idx), weights)

        indices = tf.stack([tf.range(batch_size), seqlen-1], axis=1)

        # Get only the last rnn output
        last_rnn_output = tf.nn.l2_normalize(tf.gather_nd(rnn_outputs, indices), 0)

        return input_placeholder, init_state, last_rnn_output, final_state, seqlen


def fc_layer(input, channels_in, channels_out, name):
    """Creates a fully connected neural network layer
    """
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", initializer=tf.random_normal([channels_in, channels_out]))
        biases = tf.get_variable("biases", initializer=tf.random_normal([channels_out]))
        act = tf.add(tf.matmul(input, weights), biases)
        act = tf.nn.sigmoid(act)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activations', act)

        return act


def build_optimizer(labels, output, learning_rate, optimizer='adadelta'):
    with tf.name_scope('loss'):
        losses = tf.nn.l2_loss(output - labels)
        total_loss = tf.reduce_mean(losses)
        tf.summary.scalar('cross_entropy', total_loss)
        if (optimizer == 'adagrad' or optimizer == 'sgd' or
            optimizer == 'momentum' or optimizer == 'rmsprop'):
            if optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate)
            elif optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == 'momentum':
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            elif optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(learning_rate)
            tf.summary.scalar('learning_rate', opt._learning_rate)
        else:
            if optimizer == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate)
            else:
                opt = tf.train.AdadeltaOptimizer(learning_rate)
            tf.summary.scalar('learning_rate', opt._lr)
        return opt.minimize(total_loss), total_loss


def build(embeddings, lstm_out_dimension, fc_out_dimension, max_length,
          batch_size, learning_rate, optimizer):
    """Build tensorflow graph
    """

    is_training = tf.placeholder(tf.bool)

    # Model
    with tf.variable_scope('question_embedding_rnn'):
        embeddings_placeholder = tf.placeholder(tf.float32, shape=embeddings['glove'].shape)
        embeddings_tensor = tf.Variable(embeddings_placeholder, trainable=False)

    with tf.variable_scope('lstm_layer', reuse=tf.AUTO_REUSE):
        input1, init_state1, last_rnn_output1, final_state1, seqlen1 = lstm(
            lstm_out_dimension, 2, embeddings_tensor, max_length, batch_size,
            'lstm', is_training)
        input2, init_state2, last_rnn_output2, final_state2, seqlen2 = lstm(
            lstm_out_dimension, 2, embeddings_tensor, max_length, batch_size,
            'lstm', is_training)

    lstm_output = tf.concat([tf.reshape(last_rnn_output1, [-1, lstm_out_dimension]),
                             tf.reshape(last_rnn_output2, [-1, lstm_out_dimension])],
                            1)

    tf.summary.histogram('lstm_output', lstm_output)

    fc1 = fc_layer(lstm_output, lstm_out_dimension * 2, fc_out_dimension, 'fc1')
    output = fc_layer(fc1, fc_out_dimension, 1, 'fc2')
    output = tf.reshape(output, [-1])

    # Predictions, loss, training step
    labels = tf.placeholder(tf.float32, [batch_size], name='Y')

    tf.summary.histogram('labels', labels)

    with tf.name_scope('accuracy'):
        tf.summary.histogram('output', output)
        predictions = tf.round(output)
        correct_prediction = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_step, loss = build_optimizer(labels, output, learning_rate, optimizer)

    return {
        'input_placeholders': [input1, input2],
        'labels_placeholder': labels,
        'accuracy': accuracy,
        'train_step': train_step,
        'seqlen': [seqlen1, seqlen2],
        'embeddings': embeddings_placeholder,
        'is_training': is_training,
        'loss': loss,
    }