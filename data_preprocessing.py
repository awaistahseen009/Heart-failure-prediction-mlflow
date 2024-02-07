import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class DataPreprocessing:
    def __init__(self, data_frame : pd.DataFrame, scale:bool)->None:
        self.data_frame=data_frame
        self.x=None
        self.y=None
        self.x_train=None
        self.y_train=None
        self.x_test=None
        self.y_test=None
        self.scale=scale
    def rename_cols(self)->None:
        self.data_frame.columns= ['text','target']
    
    def get_dataframe(self)->pd.DataFrame:
        return self.data_frame

    def show_shape(self)->None:
        print(f'Shape of Dataframe is: {self.data_frame.shape}')

    def show_null_values(self)->None:
        print(f'Total num values are as follows: {self.data_frame.isna().sum()}')
    
    def remove_null_values(self)->None:
        self.data_frame.dropna(inplace=True)
        print('Null values are being removed')
    
    def show_duplicates(self)->None:
        print(f'Total duplicate values are : {self.data_frame.duplicated()}')
    
    def remove_duplicates(self)->None:
        self.data_frame.drop_duplicates(inplace=True)
        print('Duplicates values are being removed')
    
    def encode_categorical_cols(self)->None:
        object_cols = self.data_frame.select_dtypes(include=['object']).columns.tolist()

        for col in object_cols:
            n=len(self.data_frame[col].unique())
            values=self.data_frame[col].unique()
            for i in range(n):
                self.data_frame[col].replace({values[i]:i},inplace=True)
    def remove_outliers(self):
        numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        for column in numerical_columns:
            Q1 = self.data_frame[column].quantile(0.25)
            Q3 = self.data_frame[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap values above upper bound
            self.data_frame[column] = self.data_frame[column].apply(lambda x: upper_bound if x > upper_bound else x)
            # Cap values below lower bound
            self.data_frame[column] = self.data_frame[column].apply(lambda x: lower_bound if x < lower_bound else x)
    def set_x_y(self , target:str):
        self.x=self.data_frame.drop([target],axis=1)
        self.y=self.data_frame[target]
    
    def scale_values(self):
        scaler=MinMaxScaler()
        self.x_train=scaler.fit_transform(self.x_train)
        self.x_test=scaler.transform(self.x_test)
    
    def split_data(self,test_size:np.float32):
        self.x_train, self.x_test,self.y_train,self.y_test=train_test_split(self.x , self.y , test_size=test_size)
        print(f'Traning has samples: {self.x_train.shape[0]} , Testing has samples: {self.x_test.shape[0]}')
    
    def get_split_data(self):
        if self.scale:
            self.scale_values()
        return self.x_train, self.x_test, self.y_train, self.y_test
    

        



        

            
