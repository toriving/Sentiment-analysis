import tensorflow as tf

class BILSTMCNN:
    def __init__(self, mode, params):
        self.hidden_dim = params['hidden_dim']        
        self.vocab_size = params['vocab_size']
        self.filter_size = params['filter_size']
        self.num_filters = params['num_filters']
        self.n_label = params['n_label']
        self.max_seq_length = params['max_seq_length']
        self.emb_dim = params['emb_dim']
        self.token_lookup = tf.get_variable('embedding', shape=[params['vocab_size'], params['emb_dim']], dtype=tf.float32)
        self.keep_prob = 1 - params['dropout_rate']
        self.max_seq_length = params['max_seq_length']
        self.d_a_size = params['d_a_size']
        self.r_size = params['r_size']
        
    def build(self, inputs, length):
        with tf.variable_scope('network'):

            embedding_token = self._make_embed(inputs)
            embed_expanded = tf.expand_dims(embedding_token, -1)

            pooled_outputs = []
            for filter_size in self.filter_size:
                pooled_outputs.append(self._conv2d_layer(embed_expanded, filter_size))

            pooled_concat = tf.concat(pooled_outputs, 3)
            conv_output = tf.reshape(pooled_concat, (-1, self.num_filters , len(self.filter_size)))
            _, states = self._rnn(embedding_token, length)
            concat_states = tf.concat([states[0].h, states[1].h], 1)
            
            logits = tf.layers.dense(concat_states, self.n_label, kernel_initializer = tf.initializers.glorot_uniform)

            predict = tf.cast(tf.argmax(logits, -1), dtype=tf.int32)

            return logits, predict


    def _conv2d_layer(self, inputs, filter_size):
        with tf.variable_scope("conv-%s" % filter_size):
            conv = tf.layers.conv2d(inputs, self.num_filters, [filter_size, self.emb_dim], padding='VALID', kernel_initializer = tf.initializers.glorot_uniform)
            conv = tf.nn.relu(conv)
            conv = tf.layers.max_pooling2d(conv, [self.max_seq_length - filter_size + 1, 1], [1, 1], padding='VALID')

        return conv

    def _make_embed(self, inputs):
        with tf.variable_scope('embedding'):
            token_embed = tf.nn.embedding_lookup(self.token_lookup, inputs)
        return token_embed

    def _rnn(self, inputs, length):
        with tf.variable_scope('rnn'):
            with tf.variable_scope("forward"):
                fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                
            with tf.variable_scope("backward"):
                bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, dtype=tf.float32)
        return outputs, state
        
