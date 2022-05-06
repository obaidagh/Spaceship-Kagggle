from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


def xgboost_model_with_random_params_search(X_train, y_train,X_val,y_val):
    
    params = {    
    'max_depth': [8,9, 10, 11, 12, 13, 14],
    'min_child_weight': [1,2, 4,8, 10],
    'colsample_bytree': [0.4, 0.6, 0.8, 1],
    'gamma': [0.0, 0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0]
    }
    
    xgb = XGBClassifier(
    learning_rate=0.0005,
    n_estimators=2000,
    reg_alpha= 0.001,
    silent=True,
    nthread=1
    )
    
    folds = 10
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    random_search = RandomizedSearchCV(
        xgb,
        param_distributions=params,
        n_iter=param_comb,
        scoring='roc_auc',
        n_jobs=4,
        cv=skf.split(X_train,y_train),
        verbose=3,
        random_state=42
    )
    
    
    random_search.fit(X_train,y_train) 
    #Training_Prediction = [temp[1] for temp in random_search.predict_proba(X_train)]
    #Validation_score = accuracy_score(y_train, Training_Prediction)
    #Prediction = [temp[1] for temp in random_search.predict_proba(X_val)]
    #Validation_score = accuracy_score(y_val, prediction)

    #return Train_scores,Validation_score,random_search
    return random_search


def use_XBG(random_search,data):
    raw_predicitoion = random_search.predict_proba(data)
    Prediction_probability=[temp[1] for temp in raw_predicitoion]

    Prediction_binary=[temp[1] >0.5 for temp in raw_predicitoion]
    
    return Prediction_probability,Prediction_binary

'''
# if you want to save the model
import joblib
#save model
joblib.dump(random_search, filename) 

#load saved model
random_search = joblib.load(filename)
'''