import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable gpu allocation log information printing.
import tensorflow as tf
import model
import data_process as dp
from configs import DEFINES
from util import check_and_create_path, BestCheckpointsExporter, get_params

def main(self):
    (train_data, train_label), (dev_data, dev_label), (test_data, test_label), _, _ = dp.data_preprocess()
    params = get_params()
    
    if DEFINES.train:
        check_and_create_path()
        
        estimator = tf.estimator.Estimator(
        model_fn=model.model_fn, 
        model_dir=DEFINES.ckpt_path,  
        params=params, config=tf.estimator.RunConfig(
              save_checkpoints_steps=30,
              save_summary_steps=1,
            log_step_count_steps=10))


        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda:dp.train_input_fn(
                train_data, train_label, DEFINES.batch_size
            ), max_steps=DEFINES.train_step)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: dp.eval_input_fn(
                dev_data, dev_label, len(dev_data)
            ), exporters = [BestCheckpointsExporter()], start_delay_secs=0, throttle_secs=0)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        print('Training finished')
        
    print('Evaluate testset')
    
    assert glob.glob(os.path.join(DEFINES.best_ckpt_path, '*.ckpt*')), 'Checkpoint does not exist'
    

    estimator = tf.estimator.Estimator(
    model_fn=model.model_fn, 
    model_dir=DEFINES.best_ckpt_path,  
    params=params)

    
    test_result = estimator.evaluate(input_fn=lambda: dp.eval_input_fn(
    test_data, test_label, len(test_data)))
    
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**test_result))



if __name__ =='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)