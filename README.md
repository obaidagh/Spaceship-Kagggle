## `Project’s title:  Spaceship Titanic-Kaggle` 
### ****Project objective:****
The main objective of the competition is to `Predict which passengers are transported to an alternate dimension`


## The project consists of 7 parts:
```
     1- Library Importing 
     2- Data Field Description and CSV Reading
     3- Exploring the Data: which is 2 parts before cleaning and after
     4- Data Cleaning and Engineering
     5- Data encoding and splitting to validation and training
     6- Models Training and validation
     7- Predicting The Test data 
```

The original dataframe contains 13 features and the target after manipulation and cleaning the final dataframe contains 14 features.
during the data cleaning:
```
     Clean_data1 function
     
     - I set the 'PassengerId' feature as the index and from it created a new feature 'GoupSize'
     - All features had about 2-2.5% missing values
     - numeric values were filled with the mean and the categorical values with the mod
     - i drop the 'Name' feature as it is not an important feature
     
     Clean_data2 function
     
     -numerical data were normalized
     - categorical data were encoded using sklearn label encoder 
     

```
     - Every feature was visualized with respect to the target using plotly library


## Models:
### 1-XGBoost classifier and hyperparameter search using scikit-learn`s random search
searchable parameters were 5:
```
    'max_depth'
    'min_child_weight'
    'colsample_bytree'
    'gamma'
    'subsample'
```
### 2- Multi-layer perceptron using Keras API
used ReduceLROnPlateau and early stopping callbacks and dropout layers

### 3- Scikit-learn library:
created a function to choose the best model of a group of models created from these classifiers
```
    a- RandomForestClassifier
    b- GradientBoostingClassifier
    c- KNeighborsClassifier
```
