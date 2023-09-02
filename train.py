import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import sys
import argparse
import json
import utils
from torch_utils import dataset, MaskedL1Loss
from model import ST_MAN
import time, datetime
import numpy as np

datasets = {"PeMS04", "PeMS08", "Loop"}

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, choices=datasets,
    default="Loop", help="use a dataset")

# get dataset argument
argv = list(sys.argv)
argv = [i for i in argv if i in datasets]
data_set = parser.parse_args(argv).dataset
config = json.load(open('data/CONFIG(%s).json' % data_set))

parser = argparse.ArgumentParser()

parser.add_argument("dataset", type=str, choices=datasets,
    default="Loop", help="use a dataset")
parser.add_argument('--time_slot', type = int, default = config['time_slot'],
                    help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = config['P'],
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = config['Q'],
                    help = 'prediction steps')
parser.add_argument('--N', type = int, default = config['N'],
                    help = 'number of Cross Att Blocks')
parser.add_argument('--L', type = int, default = config['L'],
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = config['K'],
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = config['d'],
                    help = 'dims of each head attention outputs')
parser.add_argument('--seed', type = int, default = int(time.time()),
                    help = 'seed of random utils')
parser.add_argument('--train_ratio', type = float, default = config['train_ratio'],
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = config['val_ratio'],
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = config['test_ratio'],
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = config['batch_size'],
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = config['max_epoch'],
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = config['patience'],
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = config['learning_rate'],
                    help = 'initial learning rate')
parser.add_argument('--decay_rate', type=float, default = config['decay_rate'],
                    help = 'decay rate')
parser.add_argument('--traffic_file', default = config['traffic_file'],
                    help = 'traffic file')
parser.add_argument('--SE_file', default = config['SE_file'],
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = config['model_file'],
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = config['log_file'],
                    help = 'log file')
parser.add_argument('--gpu_device', default = 'cuda:0',
                    help = 'train device')
parser.add_argument('--drop_rate', default = config['drop_rate'],
                    help = 'drop rate')
parser.add_argument('--masked_l1', default = config['masked_l1'], type=bool,
                    help = 'whether use masked l1 loss')
args = parser.parse_args()

device = torch.device(args.gpu_device if torch.cuda.is_available() else 'cpu')
log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# set seed 
seed=int(args.seed)
# utils.seed_torch(seed)

# load data
utils.log_string(log, 'loading data...')
(trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, 
mean, std) = utils.loadData(args)

trainX  = torch.FloatTensor(trainX)
valX    = torch.FloatTensor(valX)
testX   = torch.FloatTensor(testX)

trainY  = torch.FloatTensor(trainY)
valY    = torch.FloatTensor(valY)
testY   = torch.FloatTensor(testY)

SE      = torch.FloatTensor(SE).to(device)

trainTE = torch.LongTensor(trainTE)
valTE   = torch.LongTensor(valTE)
testTE  = torch.LongTensor(testTE)

utils.log_string(log, 'trainX: %s\t\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

T = 24 * 60 // args.time_slot

def train():
    train_set = dataset(trainX, trainY, trainTE, SE, device)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0)

    val_set = dataset(valX, valY, valTE, SE, device)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=0)

    utils.log_string(log, 'compiling model...')
    model = ST_MAN(1, args.P, args.Q, T, args.N, args.L, args.K, args.d, args.drop_rate, bn=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    criterion = MaskedL1Loss() if args.masked_l1 else nn.L1Loss() # whether use masked version of the `nn.L1Loss()`

    # display parameters
    parameters = 0
    for variable in model.parameters():
        parameters += np.product([x for x in variable.shape]) if variable.requires_grad else 0
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    utils.log_string(log, 'model compiled!')

    utils.log_string(log, '**** training model ****')
    # control variables
    wait = 0
    val_loss_min = np.inf

    # epoch loop
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
            break

        # train loss
        model.train()
        start_train = time.time()
        train_loss = 0.0
        # train batch loop
        for data in train_loader:
            bX, bY, bTE = data
            optimizer.zero_grad()
            p_bY = model(bX, bTE, SE)
            p_bY = p_bY * std + mean
            loss = criterion(p_bY, bY)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bX.shape[0]
        train_loss /= len(train_set)
        end_train = time.time()

        # lr decay scheduler
        scheduler.step()

        # val loss
        model.eval()
        start_val = time.time()
        val_loss = 0.0
        for data in val_loader:
            bX, bY, bTE = data
            p_bY = model(bX, bTE, SE)
            p_bY = p_bY * std + mean
            loss = criterion(p_bY, bY)
            val_loss += loss.item() * bX.shape[0]
        val_loss /= len(val_set)
        end_val = time.time()

        # epoch log
        utils.log_string(
            log,
            '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs' %
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1,
            args.max_epoch, end_train - start_train, end_val - start_val))
        utils.log_string(
            log, 'train loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
        
        if val_loss <= val_loss_min:
            utils.log_string(
                log,
                'val loss decrease from %.4f to %.4f, saving model to %s' %
                (val_loss_min, val_loss, args.model_file))
            wait = 0
            val_loss_min = val_loss
            torch.save(model.state_dict(), args.model_file)
        else:
            wait += 1


def test():
    train_set = dataset(trainX, trainY, trainTE, SE, device)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=0)

    val_set = dataset(valX, valY, valTE, SE, device)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=0)

    test_set = dataset(testX, testY, testTE, SE, device)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=0)

    utils.log_string(log, '**** testing model ****')
    utils.log_string(log, 'loading model from %s' % args.model_file)
    model = ST_MAN(1, args.P, args.Q, T, args.N, args.L, args.K, args.d, args.drop_rate, bn=True)
    model.load_state_dict(torch.load(args.model_file, device))
    model.to(device)
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    model.eval()

    # train
    trainPred = []
    for data in train_loader:
        bX, _, bTE = data
        p_bY = model(bX, bTE, SE)
        p_bY = p_bY * std + mean
        trainPred.append(p_bY.cpu().detach().numpy())
    trainPred = np.concatenate(trainPred, axis = 0)

    # val
    valPred = []
    for data in val_loader:
        bX, _, bTE = data
        p_bY = model(bX, bTE, SE)
        p_bY = p_bY * std + mean
        valPred.append(p_bY.cpu().detach().numpy())
    valPred = np.concatenate(valPred, axis = 0)
    
    # test
    start_test = time.time()
    testPred = []
    for data in test_loader:
        bX, _, bTE = data
        p_bY = model(bX, bTE, SE)
        p_bY = p_bY * std + mean
        testPred.append(p_bY.cpu().detach().numpy())
    end_test = time.time()
    testPred = np.concatenate(testPred, axis = 0)

    train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY.cpu().numpy(), args.masked_l1)
    val_mae, val_rmse, val_mape = utils.metric(valPred, valY.cpu().numpy(), args.masked_l1)
    test_mae, test_rmse, test_mape = utils.metric(testPred, testY.cpu().numpy(), args.masked_l1)

    utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
                    (train_mae, train_rmse, train_mape * 100))
    utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
                    (val_mae, val_rmse, val_mape * 100))
    utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
                    (test_mae, test_rmse, test_mape * 100))

    utils.log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    for q in range(args.Q):
        mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q].cpu().numpy(), args.masked_l1)
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        utils.log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                        (q + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    utils.log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
        (average_mae, average_rmse, average_mape * 100))



if __name__ == "__main__":
    start = time.time()
    train()
    test()
    end = time.time()
    utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()
