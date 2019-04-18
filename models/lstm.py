import sys
sys.path.append('..')
import tensorflow as tf

class LSTM:
    def __init__(self, mode, params):
        self.hidden_dim = params['hidden_dim']        
        self.vocab_size = params['vocab_size']
        self.n_label = params['n_label']
        self.token_lookup = tf.get_variable('embedding', shape=[params['vocab_size'], params['emb_dim']], dtype=tf.float32)
        self.keep_prob = 1 - params['dropout_rate']
        
    def build(self, inputs):
        with tf.variable_scope('network'):
            embedding_token = self._make_embed(inputs)
            _, rnn_state = self._rnn(embedding_token)
            logits = tf.layers.dense(rnn_state[1], self.n_label, kernel_initializer = tf.initializers.glorot_uniform) 
            predict = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        return logits, predict
             
    def _make_embed(self, inputs):
        with tf.variable_scope('embedding'):
            token_embed = tf.nn.embedding_lookup(self.token_lookup, inputs)
        return token_embed
        
    def _rnn(self, inputs):
        with tf.variable_scope('rnn'):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, initializer = tf.initializers.glorot_uniform)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        return outputs, state
        
    
