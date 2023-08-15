# ST-MAN: Spatio-Temporal Multimodal Attention Networks for Traffic Prediction

## How to Run this Model

### Data

PeMS04 and PeMS08 are provided by [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data). Seattle Loop Detector Dataset is provided by [Cui et al.](https://github.com/zhiyongc/Seattle-Loop-Data).

Our dataset files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1OAivqZcmWyYHnYcg06GQR8WSvmeNIr8B).

The Datasets files should be placed in the `data` folder.

### Requirements

Requirements should be installed before any operation.

```
pip install -r requirements.txt
```

### Manual

Run `train.py` to train the model, run `test.py` to do the test.

```
$ python train.py PeMS04
$ python test.py PeMS08
```

You should specify a dataset before trainning and testing.

```
$ python train.py --help
usage: train.py [-h] {PeMS04,PeMS08,Loop}
train.py: error: the following arguments are required: dataset
```

You can modify the settings and hyperparameters via commandline arguments, they must be placed after the `dataset` argument.

```
$ python train.py PeMS04 --help
usage: train.py [-h] [--time_slot TIME_SLOT] [--P P] [--Q Q] [--N N] [--L L]
                [--K K] [--d D] [--seed SEED] [--train_ratio TRAIN_RATIO]
                [--val_ratio VAL_RATIO] [--test_ratio TEST_RATIO]
                [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH]
                [--patience PATIENCE] [--learning_rate LEARNING_RATE]
                [--decay_rate DECAY_RATE] [--traffic_file TRAFFIC_FILE]
                [--SE_file SE_FILE] [--model_file MODEL_FILE]
                [--log_file LOG_FILE] [--gpu_device GPU_DEVICE]
                [--drop_rate DROP_RATE] [--masked_l1 MASKED_L1]
                {PeMS08,PeMS04,Loop}

positional arguments:
  {PeMS08,PeMS04,Loop}  use a dataset

optional arguments:
  -h, --help            show this help message and exit
  --time_slot TIME_SLOT
                        a time step is 5 mins
  --P P                 history steps
  --Q Q                 prediction steps
  --N N                 number of Cross Att Blocks
  --L L                 number of STAtt Blocks
  --K K                 number of attention heads
  --d D                 dims of each head attention outputs
  --seed SEED           seed of random utils
  --train_ratio TRAIN_RATIO
                        training set [default : 0.7]
  --val_ratio VAL_RATIO
                        validation set [default : 0.1]
  --test_ratio TEST_RATIO
                        testing set [default : 0.2]
  --batch_size BATCH_SIZE
                        batch size
  --max_epoch MAX_EPOCH
                        epoch to run
  --patience PATIENCE   patience for early stop
  --learning_rate LEARNING_RATE
                        initial learning rate
  --decay_rate DECAY_RATE
                        decay rate
  --traffic_file TRAFFIC_FILE
                        traffic file
  --SE_file SE_FILE     spatial emebdding file
  --model_file MODEL_FILE
                        save the model to disk
  --log_file LOG_FILE   log file
  --gpu_device GPU_DEVICE
                        train device
  --drop_rate DROP_RATE
                        drop rate
  --masked_l1 MASKED_L1
                        whether use masked l1 loss
```
