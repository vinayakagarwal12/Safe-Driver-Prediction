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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier

class Model:
    def __init__(self,cross_validation=False): 
        self.cross_validation=cross_validation
        self.df=pd.read_csv("../input/porto-seguro-new/train.csv")
        self.test=pd.read_csv("../input/porto-seguro-new/test.csv")
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

    # function to fill the null values   
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

    # function to do cross validation
    def cv_split_score(self,model):
        if(self.cross_validation):
            X_train1,X_test1,Y_train1,Y_test1=train_test_split(self.insudata,self.y,stratify=self.y,test_size=0.2,random_state=2019)
            output=[]
            if("SGD" in str(model)):
                model1=CalibratedClassifierCV(model)
                model1.fit(X_train1,Y_train1)
                output=model1.predict_proba(X_test1)[:,1]
            else:
                model.fit(X_train1,Y_train1)
                output=model.predict_proba(X_test1)[:,1]
            print(2*roc_auc_score(Y_test1,output)-1)
  

    # function to make LGBM Classifier model     
    # def GridSearchCV(self, model, params):
    # 	clf=GridSearchCV(estimator=model,param_grid=params,scoring='roc_auc',cv=5,n_jobs=-1)
	# 	return clf 
        
    # function for logistic regression  
    def logistic_regression(self):
        model = LogisticRegression(random_state=0,max_iter=1000,penalty='l2')
        return model
            

    # function for perceptron
    def perceptron(self):
        model=SGDClassifier(loss="perceptron",alpha=0.001, penalty = 'l1', n_jobs=-1)
        return model
    
    #function for svm
    def svm(self):
        model=SGDClassifier(loss="hinge",alpha=0.001, penalty = 'l1', n_jobs=-1)
        return model
    
        
    #function for random_forest_classifier
    def random_forest_classifier(self):
        model=RandomForestClassifier(n_estimators=1600,class_weight="balanced", min_samples_leaf=1000, max_leaf_nodes=150, n_jobs=-1)
        return model;



    #function for xgboost_classifier
    def xgboost_classifier(self):
        xgb_params = {}
        xgb_params['nthread']=-1
        xgb_params['learning_rate'] = 0.01
        xgb_params['n_estimators'] = 150
        # xgb_params['max_depth'] = 10
        xgb_params['eval_metric'] = 'auc'
        xgb_params['colsample_bytree'] = 0.04
        xgb_params['scale_pos_weight']=4
        model=xgb.XGBClassifier(**xgb_params)
        return model

        
    #function for lgbm classifier
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

    # function for stacking
    def stacking_classifier(self,models):
        return StackingClassifier(estimators=models,final_estimator=LogisticRegression(n_jobs=-1),stack_method='predict_proba')

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
    
    #function to create submission model output
    def create_submission_model(self,model):
        id=self.test["id"]
        self.test.drop("id",axis=1,inplace=True)
        output=self.model_predict_proba(model)
        submit1=pd.DataFrame(output)
        submit1=pd.concat([id,submit1],axis=1)
        submit1=submit1.rename(columns={0:'target'})
        submit1.to_csv("predictions.csv",index=False)



def main():
    cross_validation=True
    ml=Model(cross_validation)
    ml.fill_null_values()
    ml.one_hot_enc()
    ml.split_back()
#     model=ml.perceptron()
#     model=ml.svm()
#     model=ml.random_forest_classifier()
#     model=ml.xgboost_classifier()
    model=ml.lgbm_classifier()
    ml.cv_split_score(model)    
    # clf=ml.easy_ensemble_classifier(model)
    # ml.fit(clf)
    # ml.create_submission_model(clf)

if __name__=="__main__":
    main()