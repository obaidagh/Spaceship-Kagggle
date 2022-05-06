from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


def fitAndPredict(model,X_train, y_train,X_val,y_val):
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    
    return prediction




def model_compare(X_train, y_train,X_val,y_val):
    models=[RandomForestClassifier(n_estimators=5000, max_depth=9),
            RandomForestClassifier(n_estimators=2000, max_depth=13),
            RandomForestClassifier(n_estimators=1000, max_depth=14),
            RandomForestClassifier(n_estimators=10000, max_depth=10),
            GradientBoostingClassifier(min_samples_split=20, min_samples_leaf=60, max_depth=11, max_features=14),
            GradientBoostingClassifier(min_samples_split=10, min_samples_leaf=40, max_depth=10, max_features=14),
            GradientBoostingClassifier(min_samples_split=30, min_samples_leaf=80, max_depth=13, max_features=14),

            KNeighborsClassifier(n_neighbors=3)]
    
    traiend_models=[]
    for model in models:
        prediction=fitAndPredict(model,X_train, y_train,X_val,y_val)
        
        model_itself=[
            model,
            accuracy_score(y_val, prediction),
            confusion_matrix(y_val, prediction)
        ]
        traiend_models.append(model_itself)
        
    return traiend_models

def use_model(traiend_models,data,which_model,best_model=False):
    
    if best_model:
        which_model = [accurcy_list[1] for accurcy_list in traiend_models].index(max([accurcy_list[1] for accurcy_list in traiend_models]))
    return traiend_models[which_model][0].predict(data)

