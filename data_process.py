import tensorflow as tf
from configs import DEFINES

file_list = [DEFINES.train_data, DEFINES.dev_data, DEFINES.test_data] # original data, dataset label, sentiment label

""" Load all dataset """
def load_data(file_list):
    data = []
    for file in file_list:
        tmp = []
        with open(DEFINES.data_path + file, 'r') as f:
            tmp.extend(f.read().splitlines())
            print('number of data in %s : %d' % (file,len(tmp))) # max len = 57
        data.append(tmp)
        
    return data

def make_dataset(data):
    max_length = 0
    dataset = []
    label = []
    flag = True
    
    for ds in data:
        dataset_tmp = []
        label_tmp = []
        for d in ds:
            split_data = d.split()
            label_tmp.append(int(split_data[0]))
            text = split_data[1:]
            if max_length < len(text) and flag:
                max_length = len(text)
            
            dataset_tmp.append(text)
        dataset.append(dataset_tmp)
        label.append(label_tmp)
        flag = False
        
    print('maximum length of training data : %d' %max_length)
    return dataset, label

def make_dict(dataset):
    wtoi = dict()
    wtoi['[UNK]'] = 0
    wtoi['[PAD]'] = 1
    index = 2
    for data in dataset:
        for token in data:
            if token not in wtoi:
                wtoi[token] = index
                index += 1
    itow = {i: w for i, w in enumerate(wtoi)}
    print('the number of words : ', len(wtoi))
    
    return wtoi, itow

def word2idx(str_data, wtoi):
    output = []
    if len(str_data) < DEFINES.max_seq_length:
        str_data += ['[PAD]'] * (DEFINES.max_seq_length - len(str_data))
    else:
        str_data = str_data[:DEFINES.max_seq_length]
        
    for string in str_data:
        if string in wtoi:             
            output.append(wtoi[string])
        else:
            output.append(wtoi['[UNK]'])
    
    return output

def idx2word(idx_data, itow):
    output = []
    for idx in idx_data:
        output.append(itow[idx])
    
    return output

def dataset2idx(dataset, wtoi):
    output = []
    for data in dataset:
        output.append(word2idx(data, wtoi))
        
    return output

def data_preprocess():
    data_origin = load_data(file_list)
    
    data, label = make_dataset(data_origin)
    
    train_str, train_label = data[0], label[0]
    dev_str, dev_label = data[1], label[1]
    test_str, test_label = data[2], label[2]
    
    wtoi, itow = make_dict(train_str)
    
    train_data = dataset2idx(train_str, wtoi)
    dev_data = dataset2idx(dev_str, wtoi)
    test_data = dataset2idx(test_str, wtoi)
    
    return (train_data, train_label), (dev_data, dev_label), (test_data, test_label), wtoi, itow


def rearrange(inputs, label):
    features = {"inputs": inputs}
    return features, label


def train_input_fn(train_input, train_label, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_label))
    dataset = dataset.shuffle(buffer_size=len(train_input))
    
    assert batch_size is not None, "train batchSize must not be None"
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(DEFINES.epoch)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()


def eval_input_fn(eval_input, eval_label, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((eval_input, eval_label))
    dataset = dataset.shuffle(buffer_size=len(eval_input))
    assert batch_size is not None, "eval batchSize must not be None"
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()


# Test code
if __name__ =='__main__':
    data_origin = load_data(file_list)

    data, label = make_dataset(data_origin)

    train_data, train_label = data[0], label[0]
    dev_data, dev_label = data[1], label[1]
    test_data, test_label = data[2], label[2]

    wtoi, itow = make_dict(train_data)

    a = dataset2idx(train_data, wtoi)