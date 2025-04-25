#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import pymatgen as mg
import tqdm
from pymatgen.core import Composition
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,ConfusionMatrixDisplay,roc_curve,roc_auc_score, r2_score
from sklearn import metrics
import matplotlib.pyplot as plt

class Vectorize_Formula():

    def __init__(self,element_property_file = 'elementsnew_onehot.xlsx'):
        self.element_df = pd.read_excel(element_property_file) # CHECK NAME OF FILE 
        self.element_df.set_index('Symbol',inplace=True)
        self.column_names = []
        for column_name in list(self.element_df.columns.values[:85]):
            self.column_names.append('avg'+'_'+column_name)
        for string in ['avg','diff','max','min']:
            for column_name in list(self.element_df.columns.values[85:]):
                self.column_names.append(string+'_'+column_name)
        self.column_names_new = []
        for column_name_new in list(self.element_df.columns.values[:85]):
            self.column_names_new.append('avg'+'_'+column_name_new)
        for string_new in ['avg','diff','max','min','sum']:
            for column_name_new in list(self.element_df.columns.values[85:]):
                self.column_names_new.append(string_new+'_'+column_name_new)
        self.column_names_ratio = []
        for column_name_ratio in list(self.element_df.columns.values[:85]):
            self.column_names_ratio.append('avg'+'_'+column_name_ratio)
        for string_ratio in ['avg','diff','max','min','sum','ratio']:
            for column_name_ratio in list(self.element_df.columns.values[85:]):
                self.column_names_ratio.append(string_ratio+'_'+column_name_ratio)


    def get_features(self, formula):
        try:
            fractional_composition = Composition(formula).fractional_composition.as_dict()
            element_composition = Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                except Exception as e: 
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*(len(self.element_df.iloc[0])+3*(len(self.element_df.iloc[0])-85)))
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            
            features = np.concatenate([avg_feature, diff_feature[85:], np.array(max_feature)[85:], np.array(min_feature)[85:]])
            return features.transpose()
        except:
            print(f'There was an error with the Formula: {formula}, this is a general exception with an unkown error')
            return [np.nan]*(len(self.element_df.iloc[0])+3*(len(self.element_df.iloc[0])-85))

    def get_features_new(self, formula):
        try:
            fractional_composition = Composition(formula).fractional_composition.as_dict()
            element_composition = Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            sum_features = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                    # if self.element_df.loc[list(fractional_composition.keys())].min()==0:
                    #     feature_ratio=0
                    # else:
                    #     feature_ratio=(self.element_df.loc[list(fractional_composition.keys())].max())/(self.element_df.loc[list(fractional_composition.keys())].min())
                    sum_features += self.element_df.loc[key].values * element_composition[key]
                except Exception as e: 
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*(len(self.element_df.iloc[0])+4*(len(self.element_df.iloc[0])-85)))
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            
            features = np.concatenate([avg_feature, diff_feature[85:], np.array(max_feature)[85:], np.array(min_feature)[85:],sum_features[85:]])
            return features.transpose()
        except:
            print(f'There was an error with the Formula: {formula}, this is a general exception with an unkown error')
            return [np.nan]*(len(self.element_df.iloc[0])+4*(len(self.element_df.iloc[0])-85))

    def get_features_ratio(self, formula):
        try:
            fractional_composition = Composition(formula).fractional_composition.as_dict()
            element_composition = Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            sum_features = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                    # if self.element_df.loc[list(fractional_composition.keys())].min()==0:
                    #     feature_ratio=0
                    # else:
                    #     feature_ratio=(self.element_df.loc[list(fractional_composition.keys())].max())/(self.element_df.loc[list(fractional_composition.keys())].min())
                    sum_features += self.element_df.loc[key].values * element_composition[key]
                    ratio_features = self.element_df.loc[list(fractional_composition.keys())].max()/self.element_df.loc[list(fractional_composition.keys())].min()
                    ratio_features.replace(np.inf,1000,inplace=True)
                    ratio_features.replace(np.nan,0,inplace=True)
                except Exception as e: 
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*(len(self.element_df.iloc[0])+5*(len(self.element_df.iloc[0])-85)))
                
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            
            features = np.concatenate([avg_feature, diff_feature[85:], np.array(max_feature)[85:], np.array(min_feature)[85:],sum_features[85:],ratio_features[85:]])
            return features.transpose()
        except:
            print(f'There was an error with the Formula: {formula}, this is a general exception with an unkown error')
            return [np.nan]*(len(self.element_df.iloc[0])+5*(len(self.element_df.iloc[0])-85))
    
def write_features(data_df,
                   target_col = None,
                   label_col='Composition',
                   element_property_file='elementsnew_onehot.xlsx',
                   scale_properties=False,
                   output_file = None
                  ):
    gf=Vectorize_Formula()
    
    '''
    Takes in a pandas dataframe and outputs a dataframe of elemental and compositional properties.
    
    Parameters:
    -----------------------------------------    
    data_df: Pandas Dataframe
        columnar Dataframe with a label column and corresponding target column.
    
    label_col: string
        name of the column that contains chemical compositions
        (default = 'Composition') 
    
    target_col: string
        name of the column that contains target values
        (default = None)
        
    element_property_file: string
        path to xlsx spreadsheet with reference data for elemental properties
        (default = 'elementsnew_onehot.xlsx')
        
    scale_properties: Bool
        whether to apply a standard scaler to only the property based descriptors.
        ***WARNING***: if set to True, descriptors for predicted compounds must also be scaled by the SAME STANDARD SCALER or nothing will make sense. Leave as False unless you are sure you can scale these properly.
        (default = False)
        
    output_file: string ending in .xlsx
        file name to save the resulting spreadsheet to, will not save unless this tag is set
        (default = None)
    
   '''

    # empty lists for storage of features and targets
    features= []    

    # add values to list using for loop
    for formula in tqdm.tqdm(data_df[label_col],total=len(data_df)):
        features.append(gf.get_features(formula))
        
    # feature vectors as X
    X = pd.DataFrame(features, columns = gf.column_names)
    pd.set_option('display.max_columns', None)

    # drop elements that aren't included in the elemental properties list. 
    # These will be returned as feature rows completely full of NaN values. 
    X.dropna(inplace=True, how='all')

    # reset dataframe indices to simplify code later.
    X.reset_index(drop=True, inplace=True)

    # collect column names and find median values, fill missing values with mean
    cols = X.columns.values
    median_values = X[cols].median()
    X[cols]=X[cols].fillna(median_values.iloc[0]) #fix: drop things with missing values
    print('Data Shape:',X.shape)

    # add formation energy targets to first column
    X_cols = X.columns.tolist()
    feature_df = X[X_cols]
    
    if scale_properties == True:
        feature_df_1hot = feature_df[feature_df.columns[:85]]
        feature_df_prop = feature_df[feature_df.columns[85:]]
        
        pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('std_scaler',StandardScaler())
        ])
        feature_df_prop = pd.DataFrame(pipeline.fit_transform(feature_df_prop),columns=feature_df_prop.columns)
        feature_df = pd.concat([feature_df_1hot,feature_df_prop],axis=1)     
        
    if output_file != None:
        print('Writing output to',output_file)
        feature_df.to_excel(output_file,index=False)
    
    if target_col != None:
        outputdata_df=data_df.loc[:,target_col].to_frame()
        return feature_df,outputdata_df
    else:
        return feature_df

def write_features_new(data_df,
                   target_col = None,
                   label_col='Composition',
                   element_property_file='elementsnew_onehot.xlsx',
                   scale_properties=False,
                   output_file = None
                  ):
    gf=Vectorize_Formula()
    
    '''
    Takes in a pandas dataframe and outputs a dataframe of elemental and compositional properties.
    
    Parameters:
    -----------------------------------------    
    data_df: Pandas Dataframe
        columnar Dataframe with a label column and corresponding target column.
    
    label_col: string
        name of the column that contains chemical compositions
        (default = 'Composition') 
    
    target_col: string
        name of the column that contains target values
        (default = None)
        
    element_property_file: string
        path to xlsx spreadsheet with reference data for elemental properties
        (default = 'elementsnew_onehot.xlsx')
        
    scale_properties: Bool
        whether to apply a standard scaler to only the property based descriptors.
        ***WARNING***: if set to True, descriptors for predicted compounds must also be scaled by the SAME STANDARD SCALER or nothing will make sense. Leave as False unless you are sure you can scale these properly.
        (default = False)
        
    output_file: string ending in .xlsx
        file name to save the resulting spreadsheet to, will not save unless this tag is set
        (default = None)
    
   '''

    # empty lists for storage of features and targets
    features= []    

    # add values to list using for loop
    for formula in tqdm.tqdm(data_df[label_col],total=len(data_df)):
        features.append(gf.get_features_new(formula))
        
    # feature vectors as X
    X = pd.DataFrame(features, columns = gf.column_names_new)
    pd.set_option('display.max_columns', None)

    # drop elements that aren't included in the elemental properties list. 
    # These will be returned as feature rows completely full of NaN values. 
    X.dropna(inplace=True, how='all')

    # reset dataframe indices to simplify code later.
    X.reset_index(drop=True, inplace=True)

    # collect column names and find median values, fill missing values with mean
    cols = X.columns.values
    median_values = X[cols].median()
    X[cols]=X[cols].fillna(median_values.iloc[0]) #fix: drop things with missing values
    print('Data Shape:',X.shape)

    # add formation energy targets to first column
    X_cols = X.columns.tolist()
    feature_df = X[X_cols]
    
    if scale_properties == True:
        feature_df_1hot = feature_df[feature_df.columns[:85]]
        feature_df_prop = feature_df[feature_df.columns[85:]]
        
        pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('std_scaler',StandardScaler())
        ])
        feature_df_prop = pd.DataFrame(pipeline.fit_transform(feature_df_prop),columns=feature_df_prop.columns)
        feature_df = pd.concat([feature_df_1hot,feature_df_prop],axis=1)     
        
    if output_file != None:
        print('Writing output to',output_file)
        feature_df.to_excel(output_file,index=False)
    
    if target_col != None:
        outputdata_df=data_df.loc[:,target_col].to_frame()
        return feature_df,outputdata_df
    else:
        return feature_df

def write_features_ratio(data_df,
                   target_col = None,
                   label_col='Composition',
                   element_property_file='elementsnew_onehot.xlsx',
                   scale_properties=False,
                   output_file = None
                  ):
    gf=Vectorize_Formula()
    
    '''
    Takes in a pandas dataframe and outputs a dataframe of elemental and compositional properties.
    
    Parameters:
    -----------------------------------------    
    data_df: Pandas Dataframe
        columnar Dataframe with a label column and corresponding target column.
    
    label_col: string
        name of the column that contains chemical compositions
        (default = 'Composition') 
    
    target_col: string
        name of the column that contains target values
        (default = None)
        
    element_property_file: string
        path to xlsx spreadsheet with reference data for elemental properties
        (default = 'elementsnew_onehot.xlsx')
        
    scale_properties: Bool
        whether to apply a standard scaler to only the property based descriptors.
        ***WARNING***: if set to True, descriptors for predicted compounds must also be scaled by the SAME STANDARD SCALER or nothing will make sense. Leave as False unless you are sure you can scale these properly.
        (default = False)
        
    output_file: string ending in .xlsx
        file name to save the resulting spreadsheet to, will not save unless this tag is set
        (default = None)
    
   '''

    # empty lists for storage of features and targets
    features= []    

    # add values to list using for loop
    for formula in tqdm.tqdm(data_df[label_col],total=len(data_df)):
        features.append(gf.get_features_ratio(formula))
        
    # feature vectors as X
    X = pd.DataFrame(features, columns = gf.column_names_ratio)
    pd.set_option('display.max_columns', None)

    # drop elements that aren't included in the elemental properties list. 
    # These will be returned as feature rows completely full of NaN values. 
    X.dropna(inplace=True, how='all')

    # reset dataframe indices to simplify code later.
    X.reset_index(drop=True, inplace=True)

    # collect column names and find median values, fill missing values with mean
    cols = X.columns.values
    median_values = X[cols].median()
    X[cols]=X[cols].fillna(median_values.iloc[0]) #fix: drop things with missing values
    print('Data Shape:',X.shape)

    # add formation energy targets to first column
    X_cols = X.columns.tolist()
    feature_df = X[X_cols]
    
    if scale_properties == True:
        feature_df_1hot = feature_df[feature_df.columns[:85]]
        feature_df_prop = feature_df[feature_df.columns[85:]]
        
        pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('std_scaler',StandardScaler())
        ])
        feature_df_prop = pd.DataFrame(pipeline.fit_transform(feature_df_prop),columns=feature_df_prop.columns)
        feature_df = pd.concat([feature_df_1hot,feature_df_prop],axis=1)     
        
    if output_file != None:
        print('Writing output to',output_file)
        feature_df.to_excel(output_file,index=False)

    if target_col != None:
        outputdata_df=data_df.loc[:,target_col].to_frame()
        return feature_df,outputdata_df
    else:
        return feature_df
    
def train_rf_model(feature_df,target_df,target_range=None,mf_range=range(10,100,10),md_range=range(10,100,10),
                   num_splits=5,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
#    random.seed(1234) # turn on to make reproducible

    # define X and y
    X,y = feature_df,target_df

    # filter values out of target range
    if target_range != None:
        for target in target_df.columns:
            feature_df = feature_df[feature_df[target]>target_range[0]]
            feature_df = feature_df[feature_df[target]<target_range[1]] 
        
    # train-test split
    X_train,y_train = X.sample(frac=split_fraction,random_state=rand_state),y.sample(frac=split_fraction,random_state=rand_state)
    X_test,y_test = X.drop(X_train.index),y.drop(y_train.index)
    print('Training data size:',len(X_train))
    print('Test data size:',len(X_test))
    
    grid = GridSearchCV(RandomForestRegressor(n_estimators=n_est,n_jobs=njobs,random_state=rand_state),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='neg_mean_squared_error',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=num_splits,test_size=0.1,random_state=rand_state))
    grid.fit(X_train,y_train)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

    # Plot the score as a function of alpha
    dim1=len(mf_range)
    dim2=len(md_range)
    for i in range(dim2):
        ax1.scatter(grid.cv_results_['param_max_features'].data[i*dim1:(i+1)*dim1],
                   np.sqrt(-1*grid.cv_results_['mean_test_score'][i*dim1:(i+1)*dim1]),label=md_range[i])
    ax1.legend(loc=1)
    ax1.scatter([grid.best_params_['max_features']], [np.sqrt(-1*grid.best_score_)], marker='*', color='r', s=100)
    ax1.set_xlabel('Max. Features')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model RMSE by MF and MD')
    model = grid.best_estimator_
    
    # plot features by importance (top n_features)
    n_features = 10
    imp_df=pd.DataFrame(feature_df.columns,columns=['feature'])
    imp_df['importance']=model.feature_importances_
    imp_df=imp_df.sort_values(by=['importance'],ascending=False)
    ax2.bar(imp_df[0:n_features]['feature'],imp_df[0:n_features]['importance'])
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels(imp_df[0:n_features]['feature'],rotation=-30,ha='left')
    ax2.set_ylabel('Importance')
    ax2.set_title(f'Most Important {n_features} Features')

    

    return X_train, y_train, X_test, y_test, model, grid

def train_rf_modelbinary(feature_dfb,target_dfb,target_range=None,mf_range=range(10,100,10),md_range=range(10,100,10),
                   num_splits=5,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
    #random.seed(1234) # turn on to make reproducible

    # define X and y
    Xb,yb = feature_dfb,target_dfb

    # filter values out of target range
    if target_range != None:
        for target in target_dfb.columns:
            feature_dfb = feature_dfb[feature_dfb[target]>target_range[0]]
            feature_dfb = feature_dfb[feature_dfb[target]<target_range[1]] 
        
    # train-test split
    X_trainb,y_trainb = Xb.sample(frac=split_fraction,random_state=rand_state),yb.sample(frac=split_fraction,random_state=rand_state)
    X_testb,y_testb = Xb.drop(X_trainb.index),yb.drop(y_trainb.index)
    print('Training data size:',len(X_trainb))
    print('Test data size:',len(X_testb))
    
    gridb = GridSearchCV(RandomForestClassifier(n_estimators=n_est,n_jobs=njobs,random_state=rand_state, criterion='entropy'),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='f1',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=rand_state))
    gridb.fit(X_trainb,y_trainb)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

    # Plot the score as a function of alpha
    dim1=len(mf_range)
    dim2=len(md_range)
    for i in range(dim2):
        ax1.scatter(gridb.cv_results_['param_max_features'].data[i*dim1:(i+1)*dim1],
                   gridb.cv_results_['mean_test_score'][i*dim1:(i+1)*dim1],label=md_range[i])
    ax1.legend(loc=1)
    ax1.scatter([gridb.best_params_['max_features']], [gridb.best_score_], marker='*', color='r', s=100)
    ax1.set_xlabel('Max. Features')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model RMSE by MF and MD')
    modelb = gridb.best_estimator_
    
    # plot features by importance (top n_features)
    n_features = 10
    imp_dfb=pd.DataFrame(feature_dfb.columns,columns=['feature'])
    imp_dfb['importance']=modelb.feature_importances_
    imp_dfb=imp_dfb.sort_values(by=['importance'],ascending=False)
    ax2.bar(imp_dfb[0:n_features]['feature'],imp_dfb[0:n_features]['importance'])
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels(imp_dfb[0:n_features]['feature'],rotation=-30,ha='left')
    ax2.set_ylabel('Importance')
    ax2.set_title(f'Most Important {n_features} Features')

    return X_trainb, y_trainb, X_testb, y_testb, modelb, gridb

def test_rf_model(X_train,y_train,X_test,y_test,model,title=''):
    cv_prediction = cross_val_predict(model, X_train, y_train, cv=KFold(10, shuffle=True))   
    for scorer in ['r2_score', 'mean_absolute_error', 'mean_squared_error']:
        score = getattr(metrics,scorer)(y_train, cv_prediction)
        print('Cross validation',scorer, round(score,4))        
    predict_y = model.predict(X_test)
    mae = mean_absolute_error(y_test, predict_y)
    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test, predict_y, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlim([y_test.min(),y_test.max()])
    ax.set_ylim([y_test.min(),y_test.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Parity Plot'+title)
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(model.score(X_test,y_test),4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.5, .98, s = 'N_estimators = {0}, Max Features = {1}'.format(model.n_estimators,model.max_features),horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.5, .95, s = 'Max Depth = {0}'.format(model.max_depth), horizontalalignment='center',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def test_rf_modelboost(X_train,y_train,X_test,y_test,model):
    cv_prediction = cross_val_predict(model, X_train, y_train, cv=KFold(10, shuffle=True))   
    for scorer in ['r2_score', 'mean_absolute_error', 'mean_squared_error']:
        score = getattr(metrics,scorer)(y_train, cv_prediction)
        print('Cross validation',scorer, round(score,4))        
    predict_y = model.predict(X_test)
    mae = mean_absolute_error(y_test, predict_y)
    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test, predict_y, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlim([y_test.min()[0],y_test.max()[0]])
    ax.set_ylim([y_test.min()[0],y_test.max()[0]])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Parity Plot')
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(model.score(X_test,y_test),4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def test_rf_modelboost2(X_train,y_train,X_test,y_test,model,dim=4,title=''):
    cv_prediction = cross_val_predict(model, X_train, y_train, cv=KFold(10, shuffle=True))   
    for scorer in ['r2_score', 'mean_absolute_error']:
        score = getattr(metrics,scorer)(y_train, cv_prediction)
        print('Cross validation',scorer, round(score,4))
    print('Cross validation','root_mean_squared_error',round((getattr(metrics,'mean_squared_error')(y_train, cv_prediction))**0.5,4))
    predict_y = model.predict(X_test)
    mae = mean_absolute_error(y_test, predict_y)
    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    r2 = r2_score(y_test, predict_y)
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test, predict_y, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([0, dim], [0, dim], 'k--', lw=4)
    ax.set_xlim(0,dim)
    ax.set_ylim(0,dim)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Parity Plot {title}')
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(r2,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def test_rf_modelboost3(X_test,y_test,model,dim=4,title=''):
    # cv_prediction = cross_val_predict(model, X_train, y_train, cv=KFold(10, shuffle=True))   
    # for scorer in ['r2_score', 'mean_absolute_error']:
    #     score = getattr(metrics,scorer)(y_train, cv_prediction)
    #     print('Cross validation',scorer, round(score,4))
    # print('Cross validation','root_mean_squared_error',round((getattr(metrics,'mean_squared_error')(y_train, cv_prediction))**0.5,4))
    predict_y = model.predict(X_test)
    mae = mean_absolute_error(y_test, predict_y)
    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    r2 = r2_score(y_test, predict_y)
    ncount = len(y_test)
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test, predict_y, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([0, dim], [0, dim], 'k--', lw=4)
    ax.set_xlim(0,dim)
    ax.set_ylim(0,dim)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Parity Plot {title}')
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(r2,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .89, s = 'n = {0}'.format(ncount),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def test_rf_modelboost3root(X_test,y_test,model,dim=4,title=''):
    # cv_prediction = cross_val_predict(model, X_train, y_train, cv=KFold(10, shuffle=True))   
    # for scorer in ['r2_score', 'mean_absolute_error']:
    #     score = getattr(metrics,scorer)(y_train, cv_prediction)
    #     print('Cross validation',scorer, round(score,4))
    # print('Cross validation','root_mean_squared_error',round((getattr(metrics,'mean_squared_error')(y_train, cv_prediction))**0.5,4))
    predict_y = model.predict(X_test)
    mae = mean_absolute_error(y_test**2, predict_y**2)
    rmse = np.sqrt(mean_squared_error(y_test**2, predict_y**2))
    r2 = r2_score(y_test**2, predict_y**2)
    ncount = len(y_test)
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test**2, predict_y**2, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([0, dim], [0, dim], 'k--', lw=4)
    ax.set_xlim(0,dim)
    ax.set_ylim(0,dim)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Parity Plot {title}')
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(r2,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .89, s = 'n = {0}'.format(ncount),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def test_rf_modelbinary(modelb,X_testb,y_testb,FTlabels):
    y_predictb = modelb.predict(X_testb)
    confmatrix=confusion_matrix(y_testb, y_predictb)
    confusionchart=ConfusionMatrixDisplay(confusion_matrix = confmatrix, display_labels = FTlabels)
    confusionchart.plot()
    plt.show()
    print('Max Depth = {0}'.format(modelb.max_depth))
    print('N_estimators = {0}'.format(modelb.n_estimators))
    print('Max Features = {0}'.format(modelb.max_features))

def test_rf_modelbinary_boost(modelb2,X_testb2,y_testb2,FTlabels2):
    y_predictb2 = modelb2.predict(X_testb2)
    confmatrix2=confusion_matrix(y_testb2, y_predictb2)
    confusionchart2=ConfusionMatrixDisplay(confusion_matrix = confmatrix2, display_labels = FTlabels2)
    confusionchart2.plot()
    plt.show()
    print('Accuracy: ', round((confmatrix2[0,0]+confmatrix2[1,1])/len(X_testb2),3))
    print('Precision: ', round((confmatrix2[1,1])/sum(confmatrix2[:,1]),3))
    print('Recall: ', round((confmatrix2[1,1])/sum(confmatrix2[1,:]),3))

def parityplot(y_test,predict_y,dim=4,title=''):
    mae = mean_absolute_error(y_test, predict_y)
    rmse = np.sqrt(mean_squared_error(y_test, predict_y))
    r2 = r2_score(y_test, predict_y)
    fig, ax = plt.subplots(figsize = (7,6))
    ax.scatter(y_test, predict_y, edgecolors= (0,0,0), alpha = 0.4, color = 'grey')
    ax.plot([0, dim], [0, dim], 'k--', lw=4)
    ax.set_xlim(0,dim)
    ax.set_ylim(0,dim)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Parity Plot {title}')
    ax.text(0.02, .98, s = 'R2test: {0}'.format(round(r2,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .92, s = 'RMSE: {0}'.format(round(rmse,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    ax.text(0.02, .95, s = 'MAE: {0}'.format(round(mae,4)),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    plt.show()

def roc(model,xframe,yactual,pos=1):
    predprob=model.predict_proba(xframe)[:,pos]
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    ax1.hist(predprob)
    ax1.set_title('Probability of Positive Case')
    fpr, tpr, thresholds = roc_curve(yactual, predprob, pos_label=pos)
    roc_auc = roc_auc_score(yactual, predprob)
    ax2.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    ax2.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    ax2.set_xlabel('False Positive Rate') 
    ax2.set_ylabel('True Positive Rate') 
    ax2.set_title('ROC Curve') 
    ax2.legend(loc="lower right") 
    plt.show()

def rochelp(model,xframe,yactual,pos=1):
    predprob=model.predict_proba(xframe)[:,pos]
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    ax1.hist(predprob)
    ax1.set_title('Probability of Positive Case')
    fpr, tpr, thresholds = roc_curve(yactual, predprob, pos_label=pos)
    roc_auc = roc_auc_score(yactual, predprob)
    ax2.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    ax2.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    ax2.set_xlabel('False Positive Rate') 
    ax2.set_ylabel('True Positive Rate') 
    ax2.set_title('ROC Curve') 
    ax2.legend(loc="lower right") 
    plt.show()
    return fpr,tpr

def roc2(predprob,yactual,pos=1):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    ax1.hist(predprob)
    ax1.set_title('Probability of Positive Case')
    fpr, tpr, thresholds = roc_curve(yactual, predprob, pos_label=pos)
    roc_auc = roc_auc_score(yactual, predprob)
    ax2.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    ax2.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    ax2.set_xlabel('False Positive Rate') 
    ax2.set_ylabel('True Positive Rate') 
    ax2.set_title('ROC Curve') 
    ax2.legend(loc="lower right") 
    plt.show()

def rocconfusion(model,xframe,yactual,FTlabels2,pos=1,title=''):
    predprob=model.predict_proba(xframe)[:,pos]
    y_predictb2 = model.predict(xframe)
    confmatrix2=confusion_matrix(yactual, y_predictb2)
    confusionchart2=ConfusionMatrixDisplay(confusion_matrix = confmatrix2, display_labels = FTlabels2)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
    confusionchart2.plot(colorbar=False,ax=ax1)
    ax1.set_title(f'Confusion Matrix {title}')
    fpr, tpr, thresholds = roc_curve(yactual, predprob, pos_label=pos)
    roc_auc = roc_auc_score(yactual, predprob)
    ax2.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    ax2.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    ax2.set_xlabel('False Positive Rate') 
    ax2.set_ylabel('True Positive Rate') 
    ax2.set_title(f'ROC Curve {title}') 
    ax2.legend(loc="lower right") 
    plt.show()
    print('Accuracy: ', round((confmatrix2[0,0]+confmatrix2[1,1])/len(xframe),3))
    print('Precision: ', round((confmatrix2[1,1])/sum(confmatrix2[:,1]),3))
    print('Recall: ', round((confmatrix2[1,1])/sum(confmatrix2[1,:]),3))

def split(feature_df,target_df,split_fraction=0.8,rand_state=np.random.randint(100)):
#    random.seed(1234) # turn on to make reproducible    
    X,y = feature_df,target_df
    X_train,y_train = X.sample(frac=split_fraction,random_state=rand_state),y.sample(frac=split_fraction,random_state=rand_state)
    X_test,y_test = X.drop(X_train.index),y.drop(y_train.index)
    print('Training data size:',len(X_train))
    print('Test data size:',len(X_test))
    return X_train,y_train,X_test,y_test

def train_binaryrf_models_multiple(X_train,y_train,mf_range=range(40,50,10),md_range=range(30,40,10),
                   num_splits=3,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
#    random.seed(1234) # turn on to make reproducible
    
    grid = GridSearchCV(RandomForestClassifier(n_estimators=n_est,n_jobs=njobs,random_state=rand_state, criterion='entropy'),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='f1',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=rand_state))
    grid.fit(X_train,y_train)
    
    best=grid.best_score_

    return best

def train_rf_models_multiple(X_train,y_train,mf_range=range(40,50,10),md_range=range(30,40,10),
                   num_splits=3,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
#    random.seed(1234) # turn on to make reproducible
    
    grid = GridSearchCV(RandomForestRegressor(n_estimators=n_est,n_jobs=njobs,random_state=rand_state),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='neg_mean_squared_error',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=num_splits,test_size=0.1,random_state=rand_state))
    grid.fit(X_train,y_train)
    
    best=np.sqrt(-1*grid.best_score_)

    return best

def train_rf_model_presplit(X_train,y_train,target_range=None,mf_range=range(10,100,10),md_range=range(10,100,10),
                   num_splits=5,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
#    random.seed(1234) # turn on to make reproducible
    
    grid = GridSearchCV(RandomForestRegressor(n_estimators=n_est,n_jobs=njobs,random_state=rand_state),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='neg_mean_squared_error',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=num_splits,test_size=0.1,random_state=rand_state))
    grid.fit(X_train,y_train)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

    # Plot the score as a function of alpha
    dim1=len(mf_range)
    dim2=len(md_range)
    for i in range(dim2):
        ax1.scatter(grid.cv_results_['param_max_features'].data[i*dim1:(i+1)*dim1],
                   np.sqrt(-1*grid.cv_results_['mean_test_score'][i*dim1:(i+1)*dim1]),label=md_range[i])
    ax1.legend(loc=1)
    ax1.scatter([grid.best_params_['max_features']], [np.sqrt(-1*grid.best_score_)], marker='*', color='r', s=100)
    ax1.set_xlabel('Max. Features')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model RMSE by MF and MD')
    model = grid.best_estimator_
    
    # plot features by importance (top n_features)
    n_features = 10
    imp_df=pd.DataFrame(X_train.columns,columns=['feature'])
    imp_df['importance']=model.feature_importances_
    imp_df=imp_df.sort_values(by=['importance'],ascending=False)
    ax2.bar(imp_df[0:n_features]['feature'],imp_df[0:n_features]['importance'])
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels(imp_df[0:n_features]['feature'],rotation=-30,ha='left')
    ax2.set_ylabel('Importance')
    ax2.set_title(f'Most Important {n_features} Features')
    best=np.sqrt(-1*grid.best_score_)

    return model, grid, best

def train_rf_modelbinary_presplit(X_trainb,y_trainb,target_range=None,mf_range=range(10,100,10),md_range=range(10,100,10),
                   num_splits=5,details=0,n_est=50,njobs=-1,split_fraction=0.8,rand_state=np.random.randint(100)):
    #random.seed(1234) # turn on to make reproducible

    gridb = GridSearchCV(RandomForestClassifier(n_estimators=n_est,n_jobs=njobs,random_state=rand_state, criterion='entropy'),
                        param_grid=dict(max_features=mf_range,max_depth=md_range),
                        scoring='f1',
                        verbose=details,
                        return_train_score=(True if details>0 else False),
                        cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=rand_state))
    gridb.fit(X_trainb,y_trainb)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

    # Plot the score as a function of alpha
    dim1=len(mf_range)
    dim2=len(md_range)
    for i in range(dim2):
        ax1.scatter(gridb.cv_results_['param_max_features'].data[i*dim1:(i+1)*dim1],
                   gridb.cv_results_['mean_test_score'][i*dim1:(i+1)*dim1],label=md_range[i])
    ax1.legend(loc=1)
    ax1.scatter([gridb.best_params_['max_features']], [gridb.best_score_], marker='*', color='r', s=100)
    ax1.set_xlabel('Max. Features')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model RMSE by MF and MD')
    modelb = gridb.best_estimator_
    
    # plot features by importance (top n_features)
    n_features = 10
    imp_dfb=pd.DataFrame(X_trainb.columns,columns=['feature'])
    imp_dfb['importance']=modelb.feature_importances_
    imp_dfb=imp_dfb.sort_values(by=['importance'],ascending=False)
    ax2.bar(imp_dfb[0:n_features]['feature'],imp_dfb[0:n_features]['importance'])
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels(imp_dfb[0:n_features]['feature'],rotation=-30,ha='left')
    ax2.set_ylabel('Importance')
    ax2.set_title(f'Most Important {n_features} Features')

    return modelb, gridb

def feature_importance(X_train,model,n_features=10,title=None):
    imp_df=pd.DataFrame(X_train.columns,columns=['feature'])
    imp_df['importance']=model.feature_importances_
    imp_df=imp_df.sort_values(by=['importance'],ascending=False)
    fig, ax2 = plt.subplots(1,1,figsize=(14,6))
    ax2.bar(imp_df[0:n_features]['feature'],imp_df[0:n_features]['importance'])
    ax2.set_xlabel('Feature')
    ax2.set_xticklabels(imp_df[0:n_features]['feature'],rotation=-30,ha='left')
    ax2.set_ylabel('Importance')
    ax2.set_title(f'Most Important {n_features} Features'+title)

# def train_boostrf_modelbinary_presplit(X_trainb,y_trainb,learningrate=[0.001,0.005,0.01,0.05,0.1,0.5],
#                                        mf_range=range(0.1,1.1,0.1), maxiter=100,
#                                        num_splits=5,details=0,n_est=50,njobs=-1,minsamples=20,
#                                        split_fraction=0.8,rand_state=np.random.randint(100)):
#     #random.seed(1234) # turn on to make reproducible

#     gridb = GridSearchCV(HistGradientBoostingClassifier(min_samples_leaf=minsamples,verbose=details),
#                         param_grid=dict(learning_rate=learningrate,max_features=mf_range),
#                         scoring='f1',
#                         verbose=details,
#                         return_train_score=(True if details>0 else False),
#                         cv=ShuffleSplit(n_splits=5,test_size=0.1,random_state=rand_state)) #keep cv?
#     gridb.fit(X_trainb,y_trainb)
#     fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))

#     # Plot the score as a function of alpha
#     dim1=len(mf_range)
#     dim2=len(md_range)
#     for i in range(dim2):
#         ax1.scatter(gridb.cv_results_['param_max_features'].data[i*dim1:(i+1)*dim1],
#                    gridb.cv_results_['mean_test_score'][i*dim1:(i+1)*dim1],label=md_range[i])
#     ax1.legend(loc=1)
#     ax1.scatter([gridb.best_params_['max_features']], [gridb.best_score_], marker='*', color='r', s=100)
#     ax1.set_xlabel('Max. Features')
#     ax1.set_ylabel('RMSE')
#     ax1.set_title('Model RMSE by MF and MD')
#     modelb = gridb.best_estimator_
    
#     # plot features by importance (top n_features)
#     n_features = 10
#     imp_dfb=pd.DataFrame(X_trainb.columns,columns=['feature'])
#     imp_dfb['importance']=modelb.feature_importances_
#     imp_dfb=imp_dfb.sort_values(by=['importance'],ascending=False)
#     ax2.bar(imp_dfb[0:n_features]['feature'],imp_dfb[0:n_features]['importance'])
#     ax2.set_xlabel('Feature')
#     ax2.set_xticklabels(imp_dfb[0:n_features]['feature'],rotation=-30,ha='left')
#     ax2.set_ylabel('Importance')
#     ax2.set_title(f'Most Important {n_features} Features')

#     return modelb, gridb

# In[ ]:




