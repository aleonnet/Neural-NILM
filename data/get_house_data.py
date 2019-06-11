# -*- coding: utf-8 -*-
"""
Created on Sun Jun 2 15:52:33 2019

@author: aleonnet
"""

import pickle
from nilmtk import DataSet, TimeFrame
import pandas as pd
import numpy as np
import sklearn.datasets

# Define data path
datafolder = '/Users/alessandro/Documents/data/'
redd = DataSet(datafolder + 'redd.h5')

# Define dictionaries and sets to save data
metadata = dict(redd.metadata)
mains=[]
deviceName = []
applianced = []
appliancev = []
appliances = []
appName = []
previous_shape = (0,0)
current_shape = (0,0)
appi=0

for i in range(1,7):
    globals()['house_{}'.format(i)] = {} 

for house in range(1,7):
    # Extract data housewise
    print('house_',house)
    elec = redd.buildings[house].elec
    appliancelist = elec.appliances
    for app in appliancelist:
        previous_shape = current_shape
        label = app.type['type']
        print(' ' * 3, label)
        deviceName.append(label)
        appName = elec[label]
        data = next(appName.load(physical_quantity='power', ac_type='active')).values
#        data = next(appName.power_series()).values
        df = pd.DataFrame(data=data, index=None, dtype='float64', copy=False)
        current_shape = df.shape
        if (current_shape != previous_shape) and (appi>0):
            current_shape = previous_shape
            df = df.iloc[0:current_shape[0]]
        applianced.append(df[0])
        appliancev.append(df[0].values)
        print(current_shape[0])
        appi = appi+1
    mains = pd.concat(applianced, axis=1).sum(axis=1)
    appliances = np.array(appliancev).transpose()
    globals()['house_{}'.format(house)] = sklearn.datasets.base.Bunch(appliances=appliances, deviceName=deviceName, mains=mains.values)
    dataset = globals()['house_{}'.format(house)]
    mains = []
    deviceName = []
    applianced = []
    appliancev = []
    appliances = []
    previous_shape = (0,0)
    current_shape = (0,0)
    appi=0
    with open('house_{}.pickle'.format(house), 'wb') as file:
        # save data per house using pickle
        pickle.dump(dataset, file)
        print('End')