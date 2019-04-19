# Sentiment-analysis
An implementing of CNN and RNN for sentiment classification on SST dataset using tensorflow

## Requirements

```
pip install tensorflow-gpu==1.13.0
```


## Training the network
```
python main.py
```
or
```
python main.py \
    --batch_size=32 \
    --epoch=5 \
    --train_step=600 \
    --hidden_dim=128 \
    --vocab_size=16581 \
    --emb_dim=256 \
    --n_label=5 \
    --filter_size=[3,4,5] \
    --num_filters=100 \
    --dropout_rate=0.3 \
    --max_seq_length=20 \
    --data_path=$DATA_PATH \
    --output_path=$OUTPUT_PATH \
    --ckpt_path=$CKPT_PATH \
    --best_ckpt_path=$BEST_CKPT_PATH \
    --train_data=TRAIN_DATA \
    --dev_data=DEV_DATA \
    --test_data=TEST_DATA \
    --train=True \
    --model='LSTM'
```

## Running tests
```
python main.py --train=False --best_ckpt_path=ckpt_path
```