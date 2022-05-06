import pandas as pd,numpy as np
from sklearn.preprocessing import LabelEncoder

def Clean_data1(data):
    
    #1- split the passenger id to get the group size
    
    data[['GGGG','PP']]= data["PassengerId"].str.split("_", expand=True)
    group_size = data["GGGG"].value_counts()
    data['GroupSize'] = group_size[data["GGGG"].values].values
    data.drop(columns=['GGGG','PP'],inplace=True)
       
    #2- the passenger id is not a useful feature so use it as an index
    data.set_index('PassengerId',inplace=True)
    
    #3- split the cabin feature into 3 seperte features and convert the cabin number-which is an object- to float
      
    data[['CabinDeck','CabinNum', 'CabinSide']]= data["Cabin"].str.split("/", expand=True)
    
    data.drop(columns=['Cabin'],inplace=True)
    data['CabinNum']=data['CabinNum'].apply(pd.to_numeric).astype(float)
    
    
    #4 - fill the missing numeric features with the mean of each feature and spendings to 0 because most of passengers spendings are zero
    data['CabinNum'].fillna(value = data['CabinNum'].mean(), axis=0, inplace=True)
    data['Age'].fillna(value = data['Age'].mean(), axis=0, inplace=True)
    
    data['RoomService'].fillna(value = 0, axis=0, inplace=True)
    data['FoodCourt'].fillna(value = 0, axis=0, inplace=True)
    data['ShoppingMall'].fillna(value = 0, axis=0, inplace=True)
    data['Spa'].fillna(value = 0, axis=0, inplace=True)
    data['VRDeck'].fillna(value = 0, axis=0, inplace=True)

    
    
        
    #5 -fill the missing catagorical data 
    data['HomePlanet'].fillna(value = data['HomePlanet'].value_counts().index[0], axis=0, inplace=True)
    data['CryoSleep'].fillna(value  = data['CryoSleep'].value_counts().index[0], axis=0, inplace=True)
    data['CabinDeck'].fillna(value  = data['CabinDeck'].value_counts().index[0], axis=0, inplace=True)
    data['CabinSide'].fillna(method = 'ffill', axis=0, inplace=True)
    data['Destination'].fillna(value = data['Destination'].value_counts().index[0], axis=0, inplace=True)
    data['VIP'].fillna(value = data['VIP'].value_counts().index[0], axis=0, inplace=True)
    
    # couldn't use the mod like ---> data['HomePlanet'].fillna(value = data['HomePlanet'].mod, axis=0, inplace=True)
    # the frame would turn into method and plotly would face some problems in ploting data that are json  

    
    return data



def Clean_data2(data,normalize=False,encode_cat=False):

    #6- drop the name as it is also not that useful

    data.drop(columns=['Name'],inplace=True)
    
    #5 - optional normalization to numeric features 
    if (normalize):
        cols_to_norm = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize','CabinNum']
        data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    #7- optional cat encoding    
    if (encode_cat):
        cats = list(data.select_dtypes(exclude=['int64', 'float64']).columns)
        for i in cats:
            encoder=LabelEncoder()
            arr=np.concatenate((data[i], data[i])).astype(str)
            encoder.fit(arr)
            data[i]=encoder.fit_transform(data[i].astype(str))

    return data