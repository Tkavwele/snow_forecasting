from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scripts import utils
import os
import logging

class DataTransformation():
    def __init__(self, root):
        
        self.train_path = os.path.join(root, 'data', 'artifacts', 'train_data.csv')      
        self.val_path = os.path.join(root, 'data', 'artifacts', 'val_data.csv')  
        self.test_path = os.path.join(root, 'data', 'artifacts', 'test_data.csv')  
        self.root = root
        logging.basicConfig(level=logging.INFO)
       
    def get_data_transformer_object(self):  
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path) or not os.path.exists(self.test_path):
            logging.error("One or more data files are missing.")
            return None, None, None      
        #specify data transformation pipeline/sequence
        preprocessing_pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy='mean')),
                                         ('scaler',StandardScaler())
                                         ]
                                ) 
        return preprocessing_pipeline
    
    def initiate_data_transformation(self,
                                     ):
        preprocessor = self.get_data_transformer_object()
        
        try:
            train_df = pd.read_csv(self.train_path, index_col='date')
            val_df = pd.read_csv(self.val_path, index_col='date')
            test_df = pd.read_csv(self.test_path, index_col='date')
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None, None, None
      
        logging.info("Data loaded successfully.")
        
        target_column = 'snow'
        try:
            train_X = train_df.drop(target_column, axis=1)
            train_Y = train_df[target_column]
            val_X = val_df.drop(target_column, axis=1)
            val_Y = val_df[target_column]
            test_X = test_df.drop(target_column, axis=1)
            test_Y = test_df[target_column]
        except KeyError as e:
            logging.error(f"Missing target column: {e}")
            return None, None, None
        try:
            train_X_transformed = preprocessor.fit_transform(train_X)
            val_X_transformed = preprocessor.transform(val_X)
            test_X_transformed = preprocessor.transform(test_X)
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            return None, None, None

        # Applying data transformation
        try:
            train_X_transformed = preprocessor.fit_transform(train_X)
            val_X_transformed = preprocessor.transform(val_X)
            test_X_transformed = preprocessor.transform(test_X)
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            return None, None, None

        # Convert transformed arrays back to DataFrames with original column names
        train_X_scaled = pd.DataFrame(train_X_transformed, columns = train_X.columns)
        val_X_scaled = pd.DataFrame(val_X_transformed, columns = val_X.columns)
        test_X_scaled = pd.DataFrame(test_X_transformed, columns = test_X.columns)

        # Concatenate scaled features with target variable
        train_dataset = pd.concat([train_X_scaled.reset_index(drop=True), train_Y.reset_index(drop=True)], axis=1)
        val_dataset = pd.concat([val_X_scaled.reset_index(drop=True), val_Y.reset_index(drop=True)], axis=1)
        test_dataset = pd.concat([test_X_scaled.reset_index(drop=True), test_Y.reset_index(drop=True)], axis=1)
        
        utils.save_object(root = self.root,  
                          obj = preprocessor,
                          obj_name= 'preprocessor.pkl')
        
        logging.info("Data transformation complete and preprocessor saved.")

        return train_dataset, val_dataset, test_dataset