import sys
sys.path.append('..')
import tensorflow as tf

class BILSTM:
    def __init__(self, mode, params):
        self.hidden_dim = params['hidden_dim']        
        self.vocab_size = params['vocab_size']
        self.n_label = params['n_label']
        self.token_lookup = tf.get_variable('embedding', shape=[params['vocab_size'], params['emb_dim']], dtype=tf.float32)
        self.keep_prob = 1 - params['dropout_rate']
        
    def build(self, inputs, length):
        with tf.variable_scope('network'):
            embedding_token = self._make_embed(inputs)
            _, states = self._rnn(embedding_token, length)
            concat_states = tf.concat([states[0].h, states[1].h], 1)
            logits = tf.layers.dense(concat_states, self.n_label) 
            predict = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        return logits, predict
             
    def _make_embed(self, inputs):
        with tf.variable_scope('embedding'):
            token_embed = tf.nn.embedding_lookup(self.token_lookup, inputs)
            token_embed = tf.nn.dropout(token_embed, self.keep_prob)
        return token_embed
        
    def _rnn(self, inputs, length):
        with tf.variable_scope('rnn'):
            with tf.variable_scope("forward"):
                fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, initializer = tf.initializers.glorot_uniform)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                
            with tf.variable_scope("backward"):
                bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, initializer = tf.initializers.glorot_uniform)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length = length, dtype=tf.float32)
        return outputs, state
        
    
