#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import MatLearn_User_functions as mlf 
#need to update that python file to have "from pymatgen.core import Composition
from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mp_api.client import MPRester #import in env with "pip install mp_api"
from statistics import mean, variance
import tqdm
import itertools
import math
import pymatgen.core as mg
import time

from flatten_dict import flatten
import importlib
import reference as ref
import reference2 as ref2
from sklearn.feature_selection import VarianceThreshold,RFECV,SequentialFeatureSelector,RFE
import pandas as pd
import pymatgen as mg
import tqdm
from pymatgen.core import Composition
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,ConfusionMatrixDisplay, r2_score, f1_score,roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn import metrics
import matplotlib.pyplot as plt

import sys, json
from urllib.request import urlopen
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[3]:


def scale(dataframe):
    dataarray=np.array(dataframe)
    maxes=[]
    for i in range(len(dataarray[0])):
        maxes.append(max(list(map(abs,dataarray[:,i]))))
    arraySC=np.zeros(np.shape(dataarray))
    for i in range(len(dataarray[0])):
        for j in range(len(dataarray[:,0])):
            arraySC[j,i]=(0 if maxes[i]==0 else dataarray[j,i]/maxes[i])
    return arraySC


# In[4]:


def big_TSNE(features,color=None,text=None,n_components=50,random_state=8,n_jobs=-1,perplexity=30,alpha=0.35,scaling=True):
    if scaling==True:
        features=scale(features)
    else:
        pass
    PCA50=PCA(n_components=50)
    PCA50.fit(features)
    tfeatures = PCA50.transform(features)
    TSNERes=TSNE(random_state=8,n_jobs=-1,perplexity=perplexity).fit_transform(tfeatures)
    if color==None:
        plt.scatter(TSNERes[:,0],TSNERes[:,1])
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.text(0.025,0.95, f'perplexity={perplexity}',transform=plt.gca().transAxes)
        plt.title('t-SNE mapping')
    else:
        plt.scatter(TSNERes[:,0],TSNERes[:,1],c=color,alpha=alpha)
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.title('t-SNE mapping')
        plt.text(0.025,0.95, f'perplexity={perplexity}',transform=plt.gca().transAxes)
    if text!=None:
        spacing=0.9
        index=0
        for i in text:
            plt.text(0.025,spacing, text[index],transform=plt.gca().transAxes)
            spacing -= 0.05
            index += 1
    else:
        pass
    return TSNERes


# In[ ]:




