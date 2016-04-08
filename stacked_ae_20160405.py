# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:58:54 2016

@author: lixin

@object: Ru
"""

import pandas as pd
import numpy as np
from matplotlib import pylab
import time
import random
#import os
import gc
#import string
#import sys

###########################   loading files    #################################
data_1 = pd.read_csv('./Ru/data/FuturesData/ru201301m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_2 = pd.read_csv('./Ru/data/FuturesData/ru201302m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_3 = pd.read_csv('./Ru/data/FuturesData/ru201303m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_4 = pd.read_csv('./Ru/data/FuturesData/ru201304m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_5 = pd.read_csv('./Ru/data/FuturesData/ru201305m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_6 = pd.read_csv('./Ru/data/FuturesData/ru201306m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_7 = pd.read_csv('./Ru/data/FuturesData/ru201307m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_8 = pd.read_csv('./Ru/data/FuturesData/ru201308m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_9 = pd.read_csv('./Ru/data/FuturesData/ru201309m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_10 = pd.read_csv('./Ru/data/FuturesData/ru201310m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_11 = pd.read_csv('./Ru/data/FuturesData/ru201311m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_12 = pd.read_csv('./Ru/data/FuturesData/ru201312m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
train_data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7,
                        data_8, data_9, data_10, data_11, data_12], ignore_index = True)
del data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12

#########################################################################
#working_dir = os.path.dirname(os.path.abspath("__file__"))
#data_dir = os.path.join(os.path.pardir,'data')
##os.path.abspath(data_dir)
#dir_names = []
#for name in os.listdir(data_dir):
#    if os.path.isdir(os.path.join(data_dir,name)) and 'tick' in name:
#        dir_names.append(name)

#t1 = time.clock()
#raw_data = pd.DataFrame()
#for dir_name in dir_names:
#    for data_name in os.listdir(os.path.join(data_dir,dir_name)):
#        if data_name == 'SFIFMi.csv':
#            print 'load file from:',os.path.join(data_dir,dir_name,data_name)
#            tmp_data = pd.read_csv(os.path.join(data_dir,dir_name,data_name), 
#                                   encoding='GB18030', usecols=range(2,6)+[7]+range(12,32))
#            raw_data = pd.concat([raw_data, tmp_data], ignore_index=True)
#t2 = time.clock()
#print t2 - t1   ###  about 13 seconds
#del tmp_data
#pylab.plot(train_data[u'最新'])
#pylab.title('Last Price')
#print raw_data.columns
#######################################################################
def no_trade_dropout(raw_data):
    raw_data.columns = ["Time", "Price", "Position", "Position_chg", "Volume", 
                        "bid1", #"bid2", "bid3", "bid4", "bid5", 
                        "ask1", #"ask2", "ask3", "ask4", "ask5", 
                        "bidVOL1", #"bidVOL2", "bidVOL3", "bidVOL4", "bidVOL5", 
                        "askVOL1"]#, "askVOL2", "askVOL3", "askVOL4", "askVOL5"]
    t1 = time.clock()
    raw_data['Time'] = pd.to_datetime(raw_data['Time'])

    ##  delete days containing zero trade
    zero_index = [i for i in xrange(len(raw_data)) if raw_data['askVOL1'][i] == 0 
                  or raw_data['bidVOL1'][i] == 0]
    zero_times = [raw_data['Time'][i] for i in zero_index]
    zero_tag = []
    for zero_time in zero_times:
        month = zero_time.month
        day = zero_time.day
        if [month, day] in zero_tag:
            pass
        else:
            zero_tag.append([month, day])
    has_zero_index = [i for i in xrange(len(raw_data)) if 
                      [raw_data['Time'][i].month, raw_data['Time'][i].day] in zero_tag]

    clean_data = raw_data.drop(has_zero_index)
    clean_data.index = range(len(clean_data))
    t2 = time.clock()
    print t2 - t1
    return clean_data
    
def factorize(clean_data):
    t1 = time.clock()
    askVOL = clean_data['askVOL1']
    bidVOL = clean_data['bidVOL1']

#   ask = raw_data['ask1'] * raw_data['askVOL1'] / askVOL + raw_data['ask2'] * raw_data['askVOL2'] / askVOL + \
#         raw_data['ask3'] * raw_data['askVOL3'] / askVOL + raw_data['ask4'] * raw_data['askVOL4'] / askVOL + \
#         raw_data['ask5'] * raw_data['askVOL5'] / askVOL
#   bid = raw_data['bid1'] * raw_data['bidVOL1'] / bidVOL + raw_data['bid2'] * raw_data['bidVOL2'] / bidVOL + \
#         raw_data['bid3'] * raw_data['bidVOL3'] / bidVOL + raw_data['bid4'] * raw_data['bidVOL4'] / bidVOL + \
#         raw_data['bid5'] * raw_data['bidVOL5'] / bidVOL
    ask = clean_data['ask1']
    bid = clean_data['bid1']
    spread = ask - bid
    ave = (ask+bid)/2.0
    std_spread = np.log(spread / clean_data['Price'])
    std_ave = np.log(ave / clean_data['Price'])
    ratio = np.log(bidVOL/askVOL)
    depth = askVOL + bidVOL

    factors = pd.concat([std_spread, std_ave, clean_data['Volume'], ratio, depth,  ##  clean_data['Price'], 
                         clean_data['Position_chg']], axis = 1)  ## , raw_data['Position']
    factors.columns = ["Spread", "Ave", "Volume", "Ration", "Depth", "Position_chg"]  ## "Price",  , "Position"
    t2 = time.clock()
    print t2 - t1
    return factors

####################
def split_day(x):
    day_index = []
#    append = day_index.append
    d1 = 0
    length = len(x)
    for i in xrange(1, length):
        if x[i] != x[i-1]:
            d2 = i -1
            day_index.append([d1, d2])
#            append([d1, d2])
            d1 = i
        else:
            if i+1 == length:
                d2 = i
                if d1 != d2:
                    day_index.append([d1, d2])
#                    append([d1, d2])
    return day_index

clean_train_data = no_trade_dropout(train_data)
train_factor = factorize(clean_train_data)
#del train_data
#max(train_factor['Ration'])
t1 = time.clock()
#day_index = split_day([x.day for x in raw_data['Time']])    ## about 30 seconds
train_index = split_day([x.day for x in clean_train_data['Time']])
t2 = time.clock()
print t2-t1

#pylab.plot(raw_data['Price'])
#pylab.title('Price')
#pylab.savefig('./Ru/fig/Price.jpg',dpi=600)

def feature(x, index, change = 0.55, lag = 9):
    raise_index = []
#    raise_append = raise_index.append
    fall_index = []
#    fall_append = fall_index.append
    flat_index = []
#    flat_append = flat_index.append
    length = len(index)
    for i in xrange(length):
        for j in xrange(index[i][0]+lag+1, index[i][1]):
            if (x[j] - x[j-1]) >= change:
                raise_index.append(j-1)
#                raise_append(j-1)
            elif (x[j]-x[j-1]) <= -change:
                fall_index.append(j-1)
#                fall_append(j-1)
            elif x[j] == x[j-1]:
                flat_index.append(j-1)
#                flat_append(j-1)
    return [raise_index, fall_index, flat_index]

#def feature(x, index, change = 4e-4, lag = 9):
#    raise_index = []
#    fall_index = []
#    flat_index = []
#    for i in xrange(len(index)):
#        for j in xrange(index[i][0]+lag+1, index[i][1]):
#            if (x[j] - x[j-1])/x[j-1] >= change:
#                raise_index.append(j-1)
#            elif (x[j]-x[j-1])/x[j-1] <= -change:
#                fall_index.append(j-1)
#            elif x[j] == x[j-1]:
#                flat_index.append(j-1)
#    return [raise_index, fall_index, flat_index]
#t1 = time.clock()
#[raise_index, fall_index, flat_idx] = feature(raw_data['Price'], train_index, change=4e-4, lag=9)  ## about 200 seconds
#t2 = time.clock()
#print t2-t1

t1 = time.clock()
[raise_index, fall_index, flat_idx] = feature(clean_train_data['Price'], train_index, change=19.5, lag=9)  ## about 170 seconds
t2 = time.clock()
print t2-t1

#print clean_train_data['Time'][train_index[-1][-1]]

#t1 = time.clock()
clean_flat = []
#append = clean_idx.append
raise_set = set(raise_index)
fall_set = set(fall_index)
for i in flat_idx:
    tmp = set(range(i-9, i+1))
    if len(tmp & raise_set) == 0 and len(tmp & fall_set) == 0:
        clean_flat.append(i)
#        append(i)
#t2 = time.clock()
#print t2 - t1
del raise_set, fall_set

flat_index = random.sample(clean_flat, len(raise_index)+len(fall_index))
del clean_flat

############################  Feature Benchmark   #############################
#pylab.plot(raw_data['Price'])
#chg_raise = [raw_data['Price'][i+1]-raw_data['Price'][i] for i in raise_index]
##chg_raise = [(raw_data['Price'][i+1]-raw_data['Price'][i])/raw_data['Price'][i] for i in raise_index]
#print [max(chg_raise), min(chg_raise)]
#extrem = [i for i in xrange(len(chg_raise)) if chg_raise[i]>29]
#for idx in extrem:
#    print [raw_data['Time'][raise_index[idx]-2],raw_data['Time'][raise_index[idx]-1],
#           raw_data['Time'][raise_index[idx]], 
#           raw_data['Time'][raise_index[idx]+1], raw_data['Time'][raise_index[idx]+2],
#           raw_data['Price'][raise_index[idx]],
#           raw_data['Price'][raise_index[idx]+1]-raw_data['Price'][raise_index[idx]]]
#pylab.plot(chg_raise, marker = 'o', ms = 3)
#pylab.show()
#chg_fall = [raw_data['Price'][i+1]-raw_data['Price'][i] for i in fall_index]
##chg_fall = [(raw_data['Price'][i+1]-raw_data['Price'][i])/raw_data['Price'][i] for i in fall_index]
#print [max(chg_fall), min(chg_fall)]
#extrem = [i for i in xrange(len(chg_fall)) if chg_fall[i]<-29]
#for idx in extrem:
#    print [raw_data['Time'][fall_index[idx]-2],raw_data['Time'][fall_index[idx]-1], 
#           raw_data['Time'][fall_index[idx]],
#           raw_data['Time'][fall_index[idx]+1],raw_data['Time'][fall_index[idx]+2],
#           raw_data['Price'][fall_index[idx]],
#           raw_data['Price'][fall_index[idx]+1]-raw_data['Price'][fall_index[idx]]]
#pylab.plot(chg_fall, marker = 'o', ms = 3)
#pylab.show()
#del chg_raise, chg_fall, extrem

####################### training data check  ###########################
#raise_price = [eff_data['Price'][i] for i in raise_index]
#fall_price = [eff_data['Price'][i] for i in fall_index]
#
#raise_ave = [eff_data['Ave'][i] for i in raise_index]
#fall_ave = [eff_data['Ave'][i] for i in fall_index]
#print [max(raise_ave), min(raise_ave), max(fall_ave), min(fall_ave)]
#
#pylab.plot(raise_ave)
#pylab.title('raise')
#
#ex_raise = [i for i in xrange(len(raise_ave)) if raise_ave[i] < -0.001]
#for idx in ex_raise:
#    print raw_data['Time'][raise_index[idx]], raw_data['Price'][raise_index[idx],]
#
#pylab.plot(fall_ave)
#pylab.title('fall')
#
#pylab.plot([eff_data['Position_chg'][i] for i in raise_index])
#
#ex_ave_raise = [i for i in xrange(len(raise_ave)) if raise_ave[i] < 0]
#ex_ave_fall = [i for i in xrange(len(fall_ave)) if fall_ave[i] > 0]
#
#del raise_price, fall_price, raise_ave, fall_ave, ex_raise, ex_ave_raise, ex_ave_fall
###############################################################################
def combine(data, index, lag):
    t1 = time.clock()
    total = []
#    total_append = total.append
    for i in index:  #xrange(0, len(index))
        tmp = np.reshape(data.iloc[i-lag:i+1,:].values, (1,(np.shape(data)[1])*(lag+1)))
        total.append(tmp[0])
    t2 = time.clock()
    print t2-t1
    return pd.DataFrame(total)

raise_input = combine(data = train_factor, index=raise_index, lag = 9)
fall_input = combine(data = train_factor, index=fall_index, lag = 9)
flat_input = combine(data = train_factor, index=flat_index, lag = 9)

def normalize(x, x_max = None, x_min = None):
    if x_max is None:
        x_max = max(x)
    if x_min is None:
        x_min = min(x)
    xprime = (x - x_min) / (x_max - x_min)
    return [xprime, x_max, x_min]
    
def smooth(x, times = 3.0):
    mu = np.mean(x)
    sigma = np.std(x)
    t_1 = mu + times * sigma
    t_2 = mu - times * sigma
    length = len(x)
    for i in xrange(length):
        if x[i] > t_1:
            x[i] = t_1
        elif x[i] < t_2:
            x[i] = t_2
    return x

gc.collect()

############################  Training   ######################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, AutoEncoder#, Activation
from keras.optimizers import Adagrad#, SGD, Adadelta, RMSprop
from keras.callbacks import History#, ModelCheckpoint, EarlyStopping, Callback 
from keras.regularizers import l2#, activity_l2
from keras.utils import np_utils
#from keras import backend as K

def parse(input_data, maximums=None, minimums=None, times = 3.0, smooth_flag = True):
    _input_ = pd.DataFrame()
    for data in input_data:
        _input_ = pd.concat([_input_, pd.DataFrame(data)], ignore_index=True)
#    _input_ = pd.concat([expt_data_1, ref_data, expt_data_2])    
    input_x = []
    input_max = []
    input_min = []
    if maximums == None:
        maximums = [None for i in xrange(np.shape(_input_)[1])]
    if minimums == None:
        minimums = [None for i in xrange(np.shape(_input_)[1])]
    for i in xrange(np.shape(_input_)[1]):
        if smooth_flag is True:
            _S_input_ = smooth(_input_[i], times=times)
        else:
            _S_input_ = _input_[i]
#        _N_input_ = normalize(_input_[i], x_max=maximums[i], x_min=minimums[i])
        _N_input_ = normalize(_S_input_, x_max=maximums[i], x_min=minimums[i])
        input_x.append(_N_input_[0])
        input_max.append(_N_input_[1])
        input_min.append(_N_input_[2])
    input_x = np.array(map(np.array, zip(*input_x)))
    _y_ = []
    for i in xrange(len(input_data)):
        _y_.extend([i for j in xrange(len(input_data[i]))])
#    _y_ = np.array([1 for i in xrange(len(expt_data_1))] + [0 for i in xrange(len(ref_data))] + [2 for i in xrange(len(expt_data_2))])
    y = np_utils.to_categorical(_y_, len(input_data))
    if not any(maximums) or not any(minimums):
        return [input_x, y, input_max, input_min]
    else:
        return [input_x, y]

class sae(object):
    def __init__(self, n_sample=80, n_class=2, hidden=[200,200]):
        self.n_in = n_sample
        self.n_out = n_class
        self.hidden = hidden
        
    def train(self, train_x, train_y, batch_size=10, pre_epoch=10, 
              fine_epoch=50, dropout=0.5, regularizer_l2=1e-3, split=0.0):
        self.batch_size = batch_size
        self.p = dropout
        self.l2 = regularizer_l2
        struct = [self.n_in] + self.hidden
#        nb_epoch = pre_epoch
        
        ##  Pre-training autoencoders
        encoder = Sequential()
        encoder.add(Dense(self.hidden[0], input_dim=self.n_in, activation='tanh', W_regularizer=l2(self.l2)))
        encoder.add(Dropout(self.p))
        for n_out in self.hidden[1:]:
            encoder.add(Dense(n_out, activation='tanh', W_regularizer=l2(self.l2)))
            encoder.add(Dropout(self.p))
            
        decoder = Sequential()
        decoder.add(Dense(self.hidden[-2], input_dim=self.hidden[-1], activation='tanh', W_regularizer=l2(self.l2)))
        decoder.add(Dropout(self.p))
#        for n_in in struct[-(len(self.hidden)+1)::-1]:
        for n_in in struct[-3::-1]:
            decoder.add(Dense(n_in, activation='tanh', W_regularizer=l2(self.l2)))
            decoder.add(Dropout(self.p))
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
        ae = Sequential()
        ae.add(autoencoder)
        pre_optimizer = Adagrad(lr=0.01, epsilon=1e-6)
        ae.compile(loss='mse', optimizer=pre_optimizer)
        ae.fit(train_x, train_x, nb_epoch=pre_epoch, batch_size=self.batch_size)
        
        ##  fine-tuning
        print 'Fine-tuning the network:'
        self.model = Sequential()
        self.model.add(ae.layers[0].encoder)
        self.model.add(Dense(self.n_out, activation='softmax'))
        fine_optimizer = Adagrad(lr=0.01, epsilon=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=fine_optimizer)
#        self.early_stopping = EarlyStopping(patience=20, monitor='val_loss')
#        self.check_point = ModelCheckpoint(filepath='./tmp/weights.hdfs', save_best_only=True)
        self.history = History()
#        nb_epoch = fine_epoch
        self.model.fit(train_x, train_y, nb_epoch=fine_epoch, batch_size=self.batch_size, 
                       validation_split=split,
                       callbacks=[self.history])  ##self.check_point, ,self.early_stopping
        return self.model
        
    def predict(self, x):
        return self.model.predict_probs(x)

[input_x, input_y, input_max, input_min] = parse([flat_input, raise_input, fall_input], times=3.0, smooth_flag=True)

pd_input_max = pd.Series(input_max)
pd_input_max.to_csv('./Ru/model/input_max.csv', index = False)
pd_input_min = pd.Series(input_min)
pd_input_min.to_csv('./Ru/model/input_min.csv', index = False)

#pylab.plot(input_x[:,-8])
#pylab.savefig('./Ru/fig/position_chg_smoothed.jpg', dpi = 600)

t1 = time.clock()
model = sae(n_sample=np.shape(input_x)[1], hidden=[500,500], n_class=np.shape(input_y)[1])
model.train(input_x, input_y, batch_size=200, pre_epoch=50, fine_epoch=100000, 
            dropout=0.5, regularizer_l2=1e-5, split=0.0)
t2 = time.clock()
print t2 -t1
###############################  some plots   #############################
model.model.get_config()

LOSS =  pd.DataFrame(model.history.history['loss'])
LOSS.to_csv('./Ru/LargeEpoch_loss.csv', index = False, header = False)

pylab.plot(model.history.history['loss'])
pylab.title('loss of fine-tuning')
pylab.savefig('./Ru/fig/loss_fine_tuning.jpg', dpi = 600)
pylab.show()

score = model.model.predict(input_x)
pylab.plot(score[:,1])
pylab.title('score 1')
pylab.savefig('./Ru/fig/LargeEpoch_score_1.jpg', dpi=600)
pylab.show()

pylab.plot(score[:,2])
pylab.title('score2')
pylab.savefig('./Ru/fig/LargeEpoch_score2.jpg', dpi=600)
pylab.show()

########################  in-sample check  ####################################
train_input = []
for i in xrange(len(train_index)):
    _train_input_ = combine(data = train_factor, index = xrange(train_index[i][0]+9, train_index[i][1]), lag = 9)
    train_input.append(_train_input_)
del _train_input_

t1 = time.clock()
delta = -0.5 + 5.0 * 1
s1 = []
s2 = []
train_backtest = []
for i in xrange(len(train_input)):
#    [s1, s2] = [model_raise.predict_proba(data_normalize(train_input[i], raise_x_max, raise_x_min))[:,1],
#                model_fall.predict_proba(data_normalize(train_input[i], raise_x_max, raise_x_min))[:,1]]
#    score_train = model.model.predict(data_normalize(train_input[i], input_max, input_min))
    score_train = model.model.predict(parse([train_input[i]], input_max, input_min, smooth_flag=False)[0])
    s1.append(score_train[:,1])
    s2.append(score_train[:,2])
    train_price = list(clean_train_data["Price"][(train_index[i][0]+9):(train_index[i][1]+1)])
    train_raise_index = [j for j in xrange(np.shape(train_input[i])[0]) if score_train[j,1] > score_train[j,2]]
    train_fall_index = [j for j in xrange(np.shape(train_input[i])[0]) if score_train[j,1] < score_train[j,2]] 

#    r1 = len([m for m in train_raise_index if train_price[m+1] > train_price[m]])
#    r2 = len([m for m in train_raise_index if train_price[m+1] == train_price[m]])
#    r3 = len([m for m in train_raise_index if train_price[m+1] < train_price[m]])
#
#    f1 = len([m for m in train_fall_index if train_price[m+1] > train_price[m]])
#    f2 = len([m for m in train_fall_index if train_price[m+1] == train_price[m]])
#    f3 = len([m for m in train_fall_index if train_price[m+1] < train_price[m]])
    r1 = len([m for m in train_raise_index if (train_price[m+1] - train_price[m]) > delta])
    r2 = len([m for m in train_raise_index if abs(train_price[m+1] == train_price[m]) < delta])
    r3 = len([m for m in train_raise_index if (train_price[m+1] - train_price[m]) < -delta])

    f1 = len([m for m in train_fall_index if (train_price[m+1] - train_price[m]) > delta])
    f2 = len([m for m in train_fall_index if abs(train_price[m+1] == train_price[m]) < delta])
    f3 = len([m for m in train_fall_index if (train_price[m+1] - train_price[m]) < -delta])

    correctness = [float(r1)/(r1+f1), float(f3)/(r3+f3)]    
        
    print [[r1,r2,r3], [f1,f2,f3], correctness]
    train_backtest.append([[r1,r2,r3],[f1,f2,f3], correctness])
t2 = time.clock()
print t2 -t1

raise_correct = 0
fall_correct = 0
raise_total = 0
fall_total = 0
for i in xrange(len(train_backtest)):
    raise_correct += train_backtest[i][0][0]
    fall_correct += train_backtest[i][1][2]
    raise_total = raise_total + train_backtest[i][0][0] + train_backtest[i][1][0]
    fall_total = fall_total + train_backtest[i][0][2] + train_backtest[i][1][2]
print raise_correct, fall_correct
print float(raise_correct)/raise_total, float(fall_correct)/fall_total

f = open('./Ru/LargeEpoch.txt', 'w+')
for i in xrange(len(train_backtest)):
    f.write(str(train_backtest[i]))
    f.write("\n")
f.write(str(raise_correct))
f.write('\t')
f.write(str(float(raise_correct)/raise_total))
f.write('\n')
f.write(str(fall_correct))
f.write('\t')
f.write(str(float(fall_correct)/fall_total))
f.close()

#################################   manual prediction    ######################
weights = model.model.get_weights()
for i in xrange(len(weights)):
    pd_weight = pd.DataFrame(weights[i])
    pd_weight.to_csv('./Ru/model/weight_%i.csv' %i, index = False, header = False)
    
#_manual_data_ = combine(data = eff_data, index = xrange(test_index[0][0]+9, test_index[0][1]), lag = 9)
#manual_data = parse([_manual_data_], input_max, input_min, smooth_flag=False)[0]
#benchmark = model.model.predict(manual_data)
#pd.DataFrame(benchmark).to_csv('./Ru/model/benchmark.csv')
#print clean_data['Time'][test_index[0][0]+9]
#
#def tanh(x):
#    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#def softmax(x):
##    e = np.exp(x-np.max(x))
#    e = np.exp(x)
#    s = np.sum(e)
#    return e/s
#
#c1 = np.dot(manual_data[0], weights[0]) + weights[1]
#a1 = tanh(c1)
#c2 = np.dot(a1, weights[2]) + weights[3]
#a2 = tanh(c2)
#c3 = np.dot(a2, weights[4]) + weights[5]
#a3 = softmax(c3)
##a3.T
#def manual_predict(input_data, weights):
#    output = []   
#    for data in input_data:
#        c1 = np.dot(data, weights[0]) + weights[1]
#        a1 = tanh(c1)
#        c2 = np.dot(a1, weights[2]) + weights[3]
#        a2 = tanh(c2)
#        c3 = np.dot(a2, weights[4]) + weights[5]
#        a3 = softmax(c3)
#        output.append(a3)
#    return np.array(output)
#
#manual_test = manual_predict(input_data=manual_data, weights=weights)
#
#aa = manual_test - benchmark
#pylab.plot(aa)
##########################   threshold retrieve  ###############################
#s1 = score_train[:,1]
#s2 = score_train[:,2]

score1 = s1
score2 = s2
s1 = []
s2 = []
for s in score1:
    s1.extend(s)
for s in score2:
    s2.extend(s)
del score1, score2

print [max(s1), min(s1), len(raise_index)]
threshold_1 = 0.9998
num_1 = len([i for i in xrange(len(s1)) if (s1[i] > threshold_1)])# and s1[i] > s2[i]
print num_1

print [max(s2), min(s2), len(fall_index)]
threshold_2 = 0.9986
num_2 = len([i for i in xrange(len(s2)) if float(s2[i]) > threshold_2])
print num_2

t1 = time.clock()
delta = -0.5 + 5.0 * 4
s1 = []
s2 = []
train_backtest = []
for i in xrange(len(train_input)):
    score_train = model.model.predict(parse([train_input[i]], input_max, input_min, smooth_flag=False)[0])
    s1.append(score_train[:,1])
    s2.append(score_train[:,2])
    train_price = list(clean_train_data["Price"][(train_index[i][0]+9):(train_index[i][1]+1)])
    train_raise_index = [j for j in xrange(np.shape(train_input[i])[0]) if score_train[j,1] > threshold_1]
    train_fall_index = [j for j in xrange(np.shape(train_input[i])[0]) if score_train[j,2] > threshold_2] 

    r1 = len([m for m in train_raise_index if (train_price[m+1] - train_price[m]) > delta])
    r2 = len([m for m in train_raise_index if abs(train_price[m+1] == train_price[m]) < delta])
    r3 = len([m for m in train_raise_index if (train_price[m+1] - train_price[m]) < -delta])

    f1 = len([m for m in train_fall_index if (train_price[m+1] - train_price[m]) > delta])
    f2 = len([m for m in train_fall_index if abs(train_price[m+1] == train_price[m]) < delta])
    f3 = len([m for m in train_fall_index if (train_price[m+1] - train_price[m]) < -delta])
        
    print [[r1,r2,r3], [f1,f2,f3]]
    train_backtest.append([[r1,r2,r3],[f1,f2,f3]])
t2 = time.clock()
print t2 -t1

raise_correct = 0
fall_correct = 0
raise_total = 0
fall_total = 0
for i in xrange(len(train_backtest)):
    raise_correct += train_backtest[i][0][0]
    fall_correct += train_backtest[i][1][2]
    raise_total = raise_total + train_backtest[i][0][0] + train_backtest[i][1][0]
    fall_total = fall_total + train_backtest[i][0][2] + train_backtest[i][1][2]
print raise_correct, fall_correct

########################     some efficiency test  #######################
#a = False
#t1 = time.clock()
#for i in xrange(10000):
#    if a is False:
#        pass
#t2 = time.clock()
#print t2 - t1
#del a

############################  out of sample   ############################
data_1 = pd.read_csv('./Ru/data/FuturesData/ru201406m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))
data_2 = pd.read_csv('./Ru/data/FuturesData/ru201407m.csv',encoding='GB18030', 
                     usecols=range(2,6)+[7]+range(12,16))

test_data = pd.concat([data_1, data_2], ignore_index = True)
del data_1, data_2

#test_data = pd.read_csv('./Ru/data/FuturesData/ru201406m.csv',encoding='GB18030', 
#                     usecols=range(2,6)+[7]+range(12,16))
                     
clean_test_data = no_trade_dropout(test_data)
test_factor = factorize(clean_test_data)
t1 = time.clock()
test_index = split_day([x.day for x in clean_test_data['Time']])
t2 = time.clock()
print t2-t1

test_input = []
for i in xrange(len(test_index)):
    _test_input_ = combine(data = test_factor, index = xrange(test_index[i][0]+9, test_index[i][1]), lag = 9)
    test_input.append(_test_input_)
del _test_input_

t1 = time.clock()
test_backtest = []
for i in xrange(len(test_input)):
#    [s1, s2] = [score(test_input[i], model_raise, raise_x_max, raise_x_min),
#                score(test_input[i], model_fall, fall_x_max, fall_x_min)]
    score = model.model.predict(parse([test_input[i]], input_max, input_min, smooth_flag=False)[0])
    test_price = list(clean_test_data["Price"][(test_index[i][0]+9):(test_index[i][1]+1)])
    test_raise_index = [j for j in xrange(np.shape(test_input[i])[0]) if score[j,1] > score[j,2]]
    test_fall_index = [j for j in xrange(np.shape(test_input[i])[0]) if score[j,1] < score[j,2]] 
    
    r1 = len([m for m in test_raise_index if test_price[m+1] > test_price[m]])
    r2 = len([m for m in test_raise_index if test_price[m+1] == test_price[m]])
    r3 = len([m for m in test_raise_index if test_price[m+1] < test_price[m]])

    f1 = len([m for m in test_fall_index if test_price[m+1] > test_price[m]])
    f2 = len([m for m in test_fall_index if test_price[m+1] == test_price[m]])
    f3 = len([m for m in test_fall_index if test_price[m+1] < test_price[m]])

    correctness = [float(r1)/(r1+f1), float(f3)/(r3+f3)]    
        
    print [[r1,r2,r3], [f1,f2,f3], correctness]
    test_backtest.append([[r1,r2,r3],[f1,f2,f3], correctness])
t2 = time.clock()
print t2 - t1

raise_correct = 0
fall_correct = 0
raise_total = 0
fall_total = 0
for i in xrange(len(test_backtest)):
    raise_correct += test_backtest[i][0][0]
    fall_correct += test_backtest[i][1][2]
    raise_total = raise_total + test_backtest[i][0][0] + test_backtest[i][1][0]
    fall_total = fall_total + test_backtest[i][0][2] + test_backtest[i][1][2]
print float(raise_correct)/raise_total, float(fall_correct)/fall_total

f = open('./Ru/Ru_out_of_sample_one_model.txt', 'w+')
for i in xrange(len(test_backtest)):
    f.write(str(test_backtest[i]))
    f.write("\n")
f.write(str(float(raise_correct)/raise_total))
f.write('\n')
f.write(str(float(fall_correct)/fall_total))
f.close()

###########################   Trade Strategy   #################################
t1 = time.clock()
long_total = []
short_total = []
trade_total = []
loss = 40.0
##  should be modified later   ####

for d in xrange(len(test_index)):
#for d in [2,3]:
    #d = 0
#    test_price = list(eff_data["Price"][(test_index[d][0]+9):(test_index[d][1]+1)])
    #pylab.plot(test_price)
    long_position = False
    short_position = False
    long_index = []
    short_index = []
    long_trade = []
    short_trade = []
    trade = []
#    loss = 1.0
#    for i in xrange(len(test_price)):
    for i in xrange(test_index[d][0]+9, test_index[d][1]+1):
#        data = combine(data = eff_data, index = [i+test_index[d][0]+9], lag = 9)
        data = combine(data = test_factor, index = [i], lag = 9)
        signal = model.model.predict(parse([data], input_max, input_min, smooth_flag=False)[0])
        if long_position is False:
            if signal[0][1] > threshold_1 and signal[0][1] > signal[0][2]:
                long_position = True
    #            long_buy = test_price[i]
#                long_buy = eff_data['Price'][i+test_index[d][0]+9]
#                long_index.append(i+test_index[d][0]+9)
                long_buy = clean_test_data['Price'][i]
                long_index.append(i)
        if short_position is False:
            if signal[0][2] > threshold_2 and signal[0][2] > signal[0][1]:
                short_position = True
    #            short_buy = test_price[i]
#                short_buy = eff_data['Price'][i+test_index[d][0]+9]
#                short_index.append(i+test_index[d][0]+9)
                short_buy = clean_test_data['Price'][i]
                short_index.append(i)
        if long_position is True:
            if signal[0][2] > threshold_2 and signal[0][2] > signal[0][1]:
                long_position = False
    #            long_sell = test_price[i]  
#                long_sell = eff_data['Price'][i+test_index[d][0]+9]
                long_sell = clean_test_data['Price'][i]
                long_trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
                trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
#            if test_price[i]-long_buy <= -loss:  ## cut-off
            if clean_test_data['Price'][i]-long_buy <= -loss:
                long_position = False
    #            long_sell = test_price[i]
#                long_sell = eff_data['Price'][i+test_index[d][0]+9]
                long_sell = clean_test_data['Price'][i]
                long_trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
                trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
#            if i == len(test_price)-1:
            if i == test_index[d][1]:
                long_position = False
#                long_sell = eff_data['Price'][i+test_index[d][0]+9]
                long_sell = clean_test_data['Price'][i]
                long_trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
                trade.append({'direction':'long', 'open_time':clean_test_data['Time'][long_index[-1]], 
                                   'open_price':long_buy, 'next_tick_price':clean_test_data['Price'][long_index[-1]+1],
#                                   'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                   'close_time':clean_test_data['Time'][i],
                                   'close_price':long_sell, 'profit':long_sell-long_buy})
        if short_position is True:
            if signal[0][1] > threshold_1 and signal[0][1] > signal[0][2]:
                short_position = False
    #            short_sell = test_price[i]
#                short_sell = eff_data['Price'][i+test_index[d][0]+9]
                short_sell = clean_test_data['Price'][i]
                short_trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
                trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
            if short_buy-clean_test_data['Price'][i] <= -loss: ## cut-off
                short_position = False
    #            short_sell = test_price[i]
#                short_sell = eff_data['Price'][i+test_index[d][0]+9]
                short_sell = clean_test_data['Price'][i]
                short_trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
                trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
            if i == test_index[d][1]:
                short_position = False
#                short_sell = eff_data['Price'][i+test_index[d][0]+9]
                short_sell = clean_test_data['Price'][i]
                short_trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
                trade.append({'direction':'short', 'open_time':clean_test_data['Time'][short_index[-1]], 
                                    'open_price':short_buy, 'next_tick_price':clean_test_data['Price'][short_index[-1]+1],
#                                    'close_time':raw_data['Time'][i+test_index[d][0]+9],
                                    'close_time':clean_test_data['Time'][i],
                                    'close_price':short_sell, 'profit':short_buy-short_sell})
    
    ###### trading statistics   #########
    
    long_table = pd.DataFrame(columns= ['open_time','open_price','next_tick_price','close_time','close_price','profit','direction'])
    for i in xrange(len(long_trade)):
        long_table = pd.concat([long_table, pd.DataFrame(long_trade[i], index=[i])])
    long_table = long_table[['open_time','open_price','next_tick_price','close_time','close_price','profit','direction']]
    long_table.to_csv('./Ru/trade/long_trade_day_%i.csv' %d) 
    long_total.append(long_table)
    
    short_table = pd.DataFrame(columns= ['open_time','open_price','next_tick_price','close_time','close_price','profit','direction'])
    for i in xrange(len(short_trade)):
        short_table = pd.concat([short_table, pd.DataFrame(short_trade[i], index=[i])])
    short_table = short_table[['open_time','open_price','next_tick_price','close_time','close_price','profit','direction']]
    short_table.to_csv('./Ru/trade/short_trade_day_%i.csv' %d)  
    short_total.append(short_table)
    
    trade_table = pd.DataFrame(columns= ['open_time','open_price','next_tick_price','close_time','close_price','profit','direction'])
    for i in xrange(len(trade)):
        trade_table = pd.concat([trade_table, pd.DataFrame(trade[i], index=[i])])
    trade_table = trade_table[['open_time','open_price','next_tick_price','close_time','close_price','profit','direction']]
    trade_table.to_csv('./Ru/trade/trade_day_%i.csv' %d)  
    trade_total.append(trade_table)
t2 = time.clock()
print t2 - t1
################################   results analysis    ########################
    
def profit(trades, multiplicator = 10, interest = 0.5e-4):
    profits = []
    returns = []
    for trade in trades:
        profits.extend(trade['profit']*multiplicator-
                      (trade['open_price']+trade['close_price'])*multiplicator*interest)
        returns.extend(trade['profit']/trade['open_price'] - 
                      (trade['open_price']+trade['close_price'])*interest / trade['open_price'])
    cumulate_profits = []
    cumulate_returns = []
    for i in xrange(len(profits)):
        cumulate_profits.append(sum(profits[-(len(profits))+i::-1]))
        cumulate_returns.append(sum(returns[-(len(profits))+i::-1]))
    return [cumulate_profits, cumulate_returns]

[long_profit, long_returns] = profit(long_total, multiplicator = 10, interest = 0.5e-4)
pylab.plot(long_profit)
pylab.title('long_profits')
pylab.grid()
pylab.savefig('./Ru/fig/trade/long_l1_s3.jpg', dpi = 600)
pylab.show()
pylab.plot(long_returns)
pylab.title('long_returns')
pylab.grid()
pylab.savefig('./Ru/fig/trade/long_return_l1_s3.jpg', dpi = 600)
pylab.show()

[short_profit, short_returns] = profit(short_total, multiplicator = 10, interest = 0.5e-4)
pylab.plot(short_profit)
pylab.title('short_profits')
pylab.grid()
pylab.savefig('./Ru/fig/trade/short_l1_s3.jpg', dpi = 600)
pylab.show()
pylab.plot(short_returns)
pylab.title('short_returns')
pylab.grid()
pylab.savefig('./Ru/fig/trade/short_return_l1_s3.jpg', dpi = 600)
pylab.show()

[total_profit, total_returns] = profit(trade_total, multiplicator = 10, interest = 0.5e-4)
pylab.plot(total_profit)
pylab.title('total_profits')
pylab.grid()
pylab.savefig('./Ru/fig/trade/total_l1_s1_201210.jpg', dpi = 600)
pylab.show()
pylab.plot(total_returns)
pylab.title('total_returns')
pylab.grid()
pylab.savefig('./Ru/fig/trade/total_return_l1_s1_201210.jpg', dpi = 600)
pylab.show()

###############################################################################

def win(trades, multiplicator = 10, interest = 0.5e-4):
    r = []
    for trade in trades:
        r.extend(trade['profit']/trade['open_price'] - 
                      (trade['open_price']+trade['close_price'])*interest / trade['open_price'])
    total_trade = len(r)
    correct = [i for i in xrange(total_trade) if r[i] > 0]
    return [len(correct), total_trade, float(len(correct))/total_trade]

[long_correct, nb_long_trade, long_ratio] = win(long_total, multiplicator = 10, interest = 0.5e-4)
[short_correct, nb_short_trade, short_ratio] = win(short_total, multiplicator = 10, interest = 0.5e-4)

print [long_correct, nb_long_trade, long_ratio]
print [short_correct, nb_short_trade, short_ratio]
print float(long_correct+short_correct)/(nb_long_trade+nb_short_trade)

##########################  price in each day   ##########################
for d in xrange(len(test_index)):
    test_price = pd.Series(list(clean_test_data["Price"][(test_index[d][0]+9):(test_index[d][1]+1)]), 
                           index=clean_test_data['Time'][(test_index[d][0]+9):(test_index[d][1]+1)])
    test_price.plot()
    pylab.grid()
    pylab.savefig('./Ru/fig/test_price/Price_day_%i.jpg' %d, dpi = 600)    
    pylab.show()
    
for d in xrange(len(test_index)):
    test_price = clean_test_data.iloc[(test_index[d][0]):(test_index[d][1]+1),[0,1]]
    test_price.to_csv('./Ru/fig/test_price/Price_day_%i.csv' %d)
    
#############################   count    #################################
#n1 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]-0.2) < 0.01])
#n2 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]-0.4) < 0.01])
#n3 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]-0.6) < 0.01])
#n4 = len([i for i in xrange(len(test_price)-1) if test_price[i+1]-test_price[i] > 0.7])
#print [n1, n2, n3, n4]
#
#m1 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]+0.2) < 0.01])
#m2 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]+0.4) < 0.01])
#m3 = len([i for i in xrange(len(test_price)-1) if abs(test_price[i+1]-test_price[i]+0.6) < 0.01])
#m4 = len([i for i in xrange(len(test_price)-1) if test_price[i+1]-test_price[i] < -0.7])
#print [m1,m2,m3, m4]



































