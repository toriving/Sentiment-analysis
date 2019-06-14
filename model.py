import tensorflow as tf
from models import attbilstm, bilstm, attlstm, bilstmcnn, cnn, lstm

def model_fn(mode, features, labels, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    if not TRAIN:
        params['dropout_rate'] = 0.0
    
    if params['model'] == 'CNN':
        graph = cnn.CNN(mode, params)
    elif params['model'] == 'LSTM':
        graph = lstm.LSTM(mode, params)
    elif params['model'] == 'BILSTM':
        graph = bilstm.BILSTM(mode, params)
    elif params['model'] == 'ATTBILSTM':
        graph = attbilstm.ATTBILSTM(mode, params)
    elif params['model'] == 'ATTLSTM':
        graph = attlstm.ATTLSTM(mode, params)
    elif params['model'] == 'BILSTMCNN':
        graph = bilstmcnn.BILSTMCNN(mode, params)
    else:
        raise ValueError('Select a training model (CNN or LSTM)')
        
    logits, predict = graph.build(features['inputs'], features['length'])

    if PREDICT:
        predictions = {'indices': predict, 'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    labels_ = tf.one_hot(labels, params['n_label'])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    
    if params['model'] == ('ATTBILSTM' or 'ATTLSTM'):
        penalty = graph.penalty
        loss_P = tf.reduce_mean(penalty * params['p_coef'])
        loss = loss + loss_P
    
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict)
    
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy_train', accuracy[1])

    
    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    
    assert TRAIN, 'Select a mode'
    
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())


    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

    
    