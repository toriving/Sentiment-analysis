import sys
sys.path.append('..')
import tensorflow as tf

class ATTBILSTM:
    def __init__(self, mode, params):
        self.hidden_dim = params['hidden_dim']        
        self.vocab_size = params['vocab_size']
        self.n_label = params['n_label']
        self.token_lookup = tf.get_variable('embedding', shape=[params['vocab_size'], params['emb_dim']], dtype=tf.float32)
        self.keep_prob = 1 - params['dropout_rate']
        self.max_seq_length = params['max_seq_length']
        self.d_a_size = params['d_a_size']
        self.r_size = params['r_size']
        
    def build(self, inputs, length):
        with tf.variable_scope('network'):
            embedding_token = self._make_embed(inputs)
            outputs, _ = self._rnn(embedding_token, length)
            concat_outputs = tf.concat([outputs[0], outputs[1]], axis=2)
            att_output, self.penalty = self._self_attention(concat_outputs)
            logits = tf.layers.dense(att_output, self.n_label)
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
                fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
                
            with tf.variable_scope("backward"):
                bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
            
            outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length = length, dtype=tf.float32)
        return outputs, state
        
    def _self_attention(self, inputs):
        with tf.variable_scope('self-attention'):
            inputs_reshape = tf.reshape(inputs, [-1, 2 * self.hidden_dim])
            _H_s1 = tf.layers.dense(inputs_reshape, self.d_a_size, activation=tf.nn.tanh)
            _H_s2 = tf.layers.dense(_H_s1, self.r_size)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, self.max_seq_length, self.r_size]), [0, 2, 1])
            attention = tf.nn.softmax(_H_s2_reshape)
            penalty = self._penalization(attention)
            outputs = tf.matmul(attention, inputs)
            outputs = tf.reshape(outputs, shape=[-1, 2 * self.hidden_dim * self.r_size])
            outputs = tf.layers.dense(outputs, self.hidden_dim, activation=tf.nn.relu)
        return outputs, penalty
    
    def _penalization(self, attention):
        AA_T = tf.matmul(attention, tf.transpose(attention, [0, 2, 1]))
        I = tf.reshape(tf.tile(tf.eye(self.r_size), [tf.shape(attention)[0], 1]), [-1, self.r_size, self.r_size])
        P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))
        
        return P
    