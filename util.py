import tensorflow as tf
from configs import DEFINES
import os, shutil, glob
from collections import OrderedDict

def get_params():
    params = { 
        'batch_size': DEFINES.batch_size,
        'hidden_dim': DEFINES.hidden_dim,  
        'vocab_size': DEFINES.vocab_size,
        'n_label' : DEFINES.n_label,
        'emb_dim': DEFINES.emb_dim,
        'learning_rate': DEFINES.learning_rate,  
        'max_seq_length': DEFINES.max_seq_length,
        'filter_size': DEFINES.filter_size,
        'num_filters': DEFINES.num_filters,
        'dropout_rate': DEFINES.dropout_rate,
        'model' : DEFINES.model
    }
    return params

def check_and_create_path():
    data_out_path = os.path.join(os.getcwd(), DEFINES.output_path)
    os.makedirs(data_out_path, exist_ok=True)
    check_point_path = os.path.join(os.getcwd(),DEFINES.ckpt_path)
    os.makedirs(check_point_path, exist_ok=True)
    best_check_point_path = os.path.join(os.getcwd(),DEFINES.best_ckpt_path)
    os.makedirs(best_check_point_path, exist_ok=True)
    
class BestCheckpointsExporter(tf.estimator.BestExporter):
    
    def __init__(self, n_best=3):
        super().__init__()
        self._compare_fn = self._acc_compare_fn
        self.n_best = 3
        self.best_model = self._init_dict()
        
    def _init_dict(self):
        tmp = dict()
        for i in range(self.n_best):
            tmp[str(i)] = 0.0
        return self._sort_dict(tmp)

    def _sort_dict(self, dict):
        return OrderedDict(sorted(dict.items(), key=lambda k:k[1], reverse=True))
    
    def _acc_compare_list(self, eval_result, checkpoint_path, best_export_path):
        new_name = checkpoint_path
        new_value = eval_result['accuracy']
        
        for key, value in self.best_model.items():
            if new_value > value:
                legacy = self.best_model.popitem()[0]
                self.best_model[new_name] = new_value
                self.best_model = OrderedDict(sorted(self.best_model.items(), key=lambda k:k[1], reverse=True))
                self._delete_legacy(best_export_path, legacy)
                break
                
    def _delete_legacy(self, best_export_path, legacy):
        file_list = glob.glob(best_export_path + legacy + '*')

        for f in file_list:
            if os.path.isfile(f):
                os.remove(f)
        
    
    def _acc_compare_fn(self, curr_best_eval_result, cand_eval_result):
        default_key = "accuracy"
        
        if not curr_best_eval_result or default_key not in curr_best_eval_result:
            raise ValueError(
                'curr_best_eval_result cannot be empty or no loss is found in it.')

        if not cand_eval_result or default_key not in cand_eval_result:
            raise ValueError(
                'cand_eval_result cannot be empty or no loss is found in it.')
            
        return curr_best_eval_result[default_key] < cand_eval_result[default_key]
    
    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export, best_export_path=DEFINES.best_ckpt_path):
        
        if self._best_eval_result is None or \
                self._compare_fn(self._best_eval_result, eval_result):
            tf.logging.info(
                'Exporting a better model ({} instead of {} )...'.format(
                    eval_result, self._best_eval_result))
            
            # copy the checkpoints files *.meta *.index, *.data* each time there is a better result, no cleanup for max amount of files here
            for name in glob.glob(checkpoint_path + '.*'):
                shutil.copy(name, os.path.join(best_export_path, os.path.basename(name)))
                
           # also save the text file used by the estimator api to find the best checkpoint

            self._acc_compare_list(eval_result, os.path.basename(checkpoint_path), best_export_path)
            with open(os.path.join(DEFINES.best_ckpt_path, "checkpoint"), 'w') as f:
                f.write("model_checkpoint_path: \"{}\"".format(os.path.basename(checkpoint_path)))
                f.write("\n# best " + str(self.n_best) + " model :" + str(list(self.best_model.items())))
            print('Best ' + str(self.n_best) + ' model :', list(self.best_model.items()))
            self._best_eval_result = eval_result
        else:
            tf.logging.info(
                'Keeping the current best model ({} instead of {}).'.format(
                    self._best_eval_result, eval_result))
 