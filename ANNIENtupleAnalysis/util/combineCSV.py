import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.AmBePlots as abp
import lib.BeamPlots as bp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import scipy.misc as scm
from pylab import figure, axes, pie, title, show
from sklearn.utils import shuffle


#----------combining csv with signal CB cut & bkgd CB cut-------#
dataBKG = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Bkgd_highCB_train.csv"
dataSIGN = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_lowCB_train.csv"
newdata0 = pd.read_csv(dataSIGN,header=0).fillna(value=0)
newdata1 = pd.read_csv(dataBKG,header=0).fillna(value=0)
print(newdata0.shape,'\n',newdata1.shape)
#print('shape of BKG CSV is:', newdata1.shape, '\nThe head is\n', newdata1.head(10))
#print('\nVs shape of SIGNAL CSV is:', newdata0.shape, '\nAnd the head is\n', newdata0.head(10))
dataFINAL = pd.concat((newdata0,newdata1))
smallcsv = shuffle(dataFINAL, random_state=0)
smallcsv = smallcsv.sample(55000)
smallcsv.to_csv('small3.csv', index=False, header=False)
smallcsv.to_csv('small.csv',float_format = '%.3f', index=False, header=False)
unrealCB= dataFINAL.loc[dataFINAL['clusterChargeBalance']>1].reset_index(drop=True)



karma = smallcsv.loc[smallcsv['label']==1].reset_index(drop=True)
print('harma spape is', karma.shape)




print('unreal charge balance:\n', unrealCB.head(10))
print('unreal charge balance size:', unrealCB.shape)
#print('Shape of final dataframe is:\n', dataFINAL.shape, '\nThe head is:\n', dataFINAL.head(10), '\nThe tail is:\n', dataFINAL.tail(10))
#print('\n\ndataFINAL index\n\n', dataFINAL.index)
#dataFINAL = shuffle(dataFINAL, random_state=0)
#print('Just the tip\n', dataFINAL.sample(20))
#print('the new head is\n', dataFINAL.head())


dataFINAL.to_csv('vars_DNN_Signal_BKGD_CBcuts.csv', header=False, index=False,  float_format = '%.3f', sep=",")
dataFINAL.to_csv('labels_DNN_Signal_BKGD_CBcuts.csv', index=False,  float_format = '%.3f', sep=",")
print('head is:\n', dataFINAL.head())
newdata2 = pd.read_csv("~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_BKGD_CBcuts.csv",header=0).fillna(0)
print(newdata2.head())
print('\ndone\n')

'''
hi = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_BKGD_CBcuts.csv"
hello = pd.read_csv(hi)

print('\nTail of final dataframe is:', hello.tail())
'''
#-----------builds combined csv for evaluation-------#
evalBKG = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Bkgd_highCB_evaluate.csv"
evalSIGN = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_lowCB_evaluate.csv"

evaldata0 = pd.read_csv(evalSIGN,header=0).fillna(value=0)
evaldata1 = pd.read_csv(evalBKG,header=0).fillna(value=0)

dataEval = pd.concat((evaldata0,evaldata1))
dataEval = shuffle(dataEval, random_state=0)
print(evaldata0.shape,'\n', evaldata1.shape)
dataEval=dataEval.sample()

unrealCBeval= dataEval.loc[dataEval['clusterChargeBalance']>1].reset_index(drop=True)
print(unrealCBeval.shape)









