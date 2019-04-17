import tensorflow as tf

class CNN:
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
        
    def build(self, inputs):
        with tf.variable_scope('network'):

            embedding_token = self._make_embed(inputs)
            embed_expanded = tf.expand_dims(embedding_token, -1)

            pooled_outputs = []
            for filter_size in self.filter_size:
                pooled_outputs.append(self._conv2d_layer(embed_expanded, filter_size))

            pooled_concat = tf.concat(pooled_outputs, 3)
            conv_output = tf.reshape(pooled_concat, (-1, self.num_filters * len(self.filter_size)))
            dropout_conv = tf.nn.dropout(conv_output, self.keep_prob)

            logits = tf.layers.dense(dropout_conv, self.n_label)

            predict = tf.cast(tf.argmax(logits, -1), dtype=tf.int32)

            return logits, predict


    def _conv2d_layer(self, inputs, filter_size):
        with tf.variable_scope("conv-%s" % filter_size):
            conv = tf.layers.conv2d(inputs, self.num_filters, [filter_size, self.emb_dim], padding='VALID')
            conv = tf.nn.relu(conv)
            conv = tf.layers.max_pooling2d(conv, [self.max_seq_length - filter_size + 1, 1], [1, 1], padding='VALID')

        return conv

    def _make_embed(self, inputs):
        with tf.variable_scope('embedding'):
            token_embed = tf.nn.embedding_lookup(self.token_lookup, inputs)
        return token_embed
