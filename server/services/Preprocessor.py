import pickle

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self, model_features, model_type, scaler_path):
        '''
        PARAMETERS:
        ------
        `model_features`: list of feature names that are used in the model
        '''
        self.model_features = model_features
        self.model_type = model_type
    
        # Load the saved scaler
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

    def create_encoded_features(self, features):
        if self.model_type == 'ffnn':
            return self.create_encoded_features_FFNet(features)
        elif self.model_type == 'rf':
            return self.create_encoded_features_RF(features)
        elif self.model_type == 'svm':
            return self.create_encoded_features_SVM(features)
        else:
            raise Exception('Model type not supported')
    
    def create_encoded_features_FFNet(self, features):
        # Transform features into a pandas dataframe
        df = pd.DataFrame(features, index=[0])
        # Encode features
        df = df.replace({True: 1, False: 0})
        df = df.replace('10.000.000.000', 1.0)
        
        # Remove features that are not in the model
        df = self.remove_features(df)
        # Scale the features
        df = self.scale_features(df)
        # Convert to tensor
        tensor = torch.FloatTensor(df)

        return tensor
    
    def create_encoded_features_RF(self, features):
        # Transform features into a pandas dataframe
        df = pd.DataFrame(features, index=[0])
        # Encode features
        df = df.replace({True: 1, False: 0})
        df = df.replace('10.000.000.000', 1.0)

        # Remove features that are not in the model
        df = self.remove_features(df)

        # Scale the features
        df = self.scale_features(df)

        return df
    
    def create_encoded_features_SVM(self, features):
        data = pd.DataFrame(features, index=[0])
        
        data = self.remove_features(data)
        data = self.scale_features(data)
        return data
            
    def scale_features(self, df):
        # Scale the features
        df = self.scaler.transform(df)
        return df

    def remove_features(self, df):
        return df[self.model_features]

def test_preprocessor():

    model_features = [
        "NumDots",
        "SubdomainLevel",
        "PathLevel",    
        "UrlLength",
        "NumDash",
        "NumDashInHostname",
        "AtSymbol",
        "TildeSymbol",
        "NumUnderscore",
        "NumPercent",
        "NumQueryComponents",
        "NumAmpersand",
        "NumHash",
        "NumNumericChars",
        "IpAddress",
        "DomainInSubdomains",
        "DomainInPaths",
        "HttpsInHostname",
        "HostnameLength",
        "PathLength",
        "QueryLength",
        "DoubleSlashInPath",
        "NumSensitiveWords",
        "PctExtResourceUrls",
        "InsecureForms",
        "ExtFormAction",
        "PopUpWindow",
        "SubmitInfoToEmail",
        "IframeOrFrame",
        "MissingTitle",
        "ImagesOnlyInForm",
    ]
    preprocessor = Preprocessor(model_features=model_features, model_type='FFNet', scaler_path='./exp/scaler.pkl')
    features = {
        "NumDots": 5,
        "SubdomainLevel": 2,
        "PathLevel": 2,
        "UrlLength": 123,
        "NumDash": 3,
        "NumDashInHostname": 1,
        "AtSymbol": True,
        "TildeSymbol": False,
        "NumUnderscore": 1,
        "NumPercent": 0,
        "NumQueryComponents": 2,
        "NumAmpersand": 0,
        "NumHash": 0,
        "NumNumericChars": 2,
        "IpAddress": False,
        "DomainInSubdomains": True,
        "DomainInPaths": False,
        "HttpsInHostname": True,
        "HostnameLength": 6,
        "PathLength": 5,
        "QueryLength": 12,
        "DoubleSlashInPath": False,
        "NumSensitiveWords": 2,
        "PctExtResourceUrls": 0.2,
        "InsecureForms": True,
        "ExtFormAction": True,
        "PopUpWindow": True,
        "SubmitInfoToEmail": False,
        "IframeOrFrame": False,
        "MissingTitle": False,
        "ImagesOnlyInForm": False,
        "test": False
    }
    preprocessor.create_encoded_features(features)

if __name__ == "__main__":
    test_preprocessor()