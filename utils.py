import os
import random
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set Seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label, masked: bool):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        if masked:
            mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        if masked:
            rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance_TE(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def seq2instance(data, P, Q):
    # print(data.shape)
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, 1))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        tmp = data[i + P : i + P + Q,:,0]
        tmp = tmp.reshape(tmp.shape + (1,))
        y[i] = tmp
    return x, y

def loadData(args):
    # Traffic
    if args.dataset in {'PeMS04', 'PeMS08', 'Loop'}:
        df = np.load(args.traffic_file, allow_pickle=True)
        Traffic = df['data']
    else:
        df = pd.read_hdf(args.traffic_file)
        Traffic = df.values
    # train/val/test
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    mean, std = np.mean(trainX,axis=(0,1,2),keepdims=True), np.std(trainX, axis=(0,1,2), keepdims=True)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std
    mean, std = mean[0,0,0,0], std[0,0,0,0]

    # spatial embedding 
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]
        
    # temporal embedding 
    if args.dataset in {'PeMS04', 'PeMS08'}:
        start_time = datetime(
            2016, 7, 1, 0, 0, 0) if args.dataset == 'PeMS08' else datetime(2018, 1, 1, 0, 0, 0)
        Time = [start_time + i * timedelta(minutes=5) for i in range(num_step)]
        Time = pd.to_datetime(Time)
    elif args.dataset == 'Loop':
        Time = df['time']
        Time = pd.to_datetime(Time)
    else:
        Time = df.index
    
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (300.0) # (<5 * Minutes>.total_seconds())  # Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    # Time Series 6&8
    Peri_72 = timeofday//72
    Peri_96 = timeofday//96
    Peri_144 = timeofday//144
    Time = np.concatenate((dayofweek, timeofday, Peri_72, Peri_96, Peri_144), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance_TE(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance_TE(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance_TE(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)
