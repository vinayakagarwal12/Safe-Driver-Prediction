#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.utils import resample
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from imblearn.ensemble import EasyEnsembleClassifier
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--train_set")
parser.add_argument("--test_set")
args = parser.parse_args() 
class Model:
    def __init__(self): 
        self.df=pd.read_csv(args.train_set)
        self.test=pd.read_csv(args.test_set)
        self.y=self.df['target']
        self.df.drop('target',axis=1,inplace=True)
        self.insudata=pd.concat([self.test.assign(ind="test"), self.df.assign(ind="train")],axis=0,ignore_index=True)
        self.insudata=self.insudata.replace(to_replace=-1,value=np.nan)
        self.calc=[]
        for i in self.insudata.columns:
            if "calc" in i and "bin" not in i and i!="ps_calc_02":
                self.calc.append(i)
        self.binary=[]
        self.cat=[]
        self.rest=[]
        self.insudata.drop(["ps_ind_10_bin","ps_ind_13_bin"],axis=1,inplace=True)
        self.insudata.drop(self.calc,axis=1,inplace=True)
        for i in self.insudata.columns:
            if( i[-3:]=="bin"):
                self.binary.append(i)
            elif( i[-3:]=="cat"):
                self.cat.append(i)
            else:
                self.rest.append(i)
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None

    def fill_null_values(self):
        insudata=self.insudata
        insudata["ps_ind_02_cat"]=insudata["ps_ind_02_cat"].fillna(insudata["ps_ind_02_cat"].mode()[0])
        insudata["ps_ind_04_cat"]=insudata["ps_ind_04_cat"].fillna(insudata["ps_ind_04_cat"].mode()[0])
        insudata["ps_ind_05_cat"]=insudata["ps_ind_05_cat"].fillna(insudata["ps_ind_05_cat"].mode()[0])
        insudata["ps_car_01_cat"]=insudata["ps_car_01_cat"].fillna(insudata["ps_car_01_cat"].mode()[0])
        insudata["ps_car_02_cat"]=insudata["ps_car_02_cat"].fillna(insudata["ps_car_02_cat"].mode()[0])
        insudata["ps_car_07_cat"]=insudata["ps_car_07_cat"].fillna(insudata["ps_car_07_cat"].mode()[0])
        insudata["ps_car_09_cat"]=insudata["ps_car_09_cat"].fillna(insudata["ps_car_09_cat"].mode()[0])
        insudata["ps_car_03_cat"]=insudata["ps_car_03_cat"].fillna(2.0)
        insudata["ps_car_05_cat"]=insudata["ps_car_05_cat"].fillna(2.0)
        insudata["ps_reg_03"]=insudata["ps_reg_03"].fillna(insudata["ps_reg_03"].mean())
        insudata["ps_car_11"]=insudata["ps_car_11"].fillna(insudata["ps_car_11"].mean())
        insudata["ps_car_14"]=insudata["ps_car_14"].fillna(insudata["ps_car_14"].mean())
        insudata["ps_car_12"]=insudata["ps_car_12"].fillna(insudata["ps_car_12"].mean())
        self.insudata=insudata
    
    
    # function to do One Hot Encoding of categorical features    
    def one_hot_enc(self):
        l=pd.concat([self.insudata[self.binary],self.insudata[self.rest]],axis=1)
        enc_one_hot=OneHotEncoder(handle_unknown='ignore',sparse=True)
        cat_tp=pd.DataFrame()
        for i in self.cat:
            enc_df = enc_one_hot.fit_transform(self.insudata[[i]]).toarray()
            names=enc_one_hot.get_feature_names([i])
            enc_df=pd.DataFrame(enc_df,columns=names)
            cat_tp=pd.concat([cat_tp,enc_df],axis=1)
        l=pd.concat([l,cat_tp],axis=1)
        return l
    
    # function to split the merged data back to train and test data
    def split_back(self):
        l=self.one_hot_enc()
        self.test= l[l["ind"].eq("test")].copy()
        self.insudata=l[l["ind"].eq("train")].copy()
        self.test.drop("ind",axis=1,inplace=True)
        self.insudata.drop("ind",axis=1,inplace=True)
        self.insudata.reset_index(drop=True)
        self.insudata.drop(["id"],axis=1,inplace=True)
        self.rest.remove("id")

    # function to make LGBM Classifier model     
    def lgbm_classifier(self):
        params={
                'objective':'binary',
                'n_estimators':1600,
                'boosting_type':'goss',
                'n_jobs':-1,
                'col_bysample':0.04,
                'learning_rate':0.005,
                'max_bin':10,
                }
        model= LGBMClassifier(**params)
        return model

    # function to make Easy Ensemble Classifier model
    def easy_ensemble_classifier(self,model):
        clf=EasyEnsembleClassifier(n_estimators=45,base_estimator=model,random_state=42,n_jobs=-1,sampling_strategy='majority')
        return clf 
    
    # function to train a model
    def fit(self,model):
        model.fit(self.insudata,self.y)

    # function to return the output probalities predicted by model
    def model_predict_proba(self,model):
        output=model.predict_proba(self.test)[:,1]
        return output

    def create_submission_model(self,model):
        id=self.test["id"]
        self.test.drop("id",axis=1,inplace=True)
        output=self.model_predict_proba(model)
        submit1=pd.DataFrame(output)
        submit1=pd.concat([id,submit1],axis=1)
        submit1=submit1.rename(columns={0:'target'})
        submit1.to_csv("submission.csv",index=False)



def main():
    ml=Model()
    ml.fill_null_values()
    ml.one_hot_enc()
    ml.split_back()
    model=ml.lgbm_classifier()
    clf=ml.easy_ensemble_classifier(model)
    ml.fit(clf)
    ml.create_submission_model(clf)

if __name__=="__main__":
    main()
