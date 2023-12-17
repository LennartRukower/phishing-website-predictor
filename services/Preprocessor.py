import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


class Preprocessor:

    def __init__(self, model_features, model_type):
        '''
        PARAMETERS:
        ------
        `model_features`: list of feature names that are used in the model
        '''
        self.model_features = model_features
        self.model_type = model_type
        if self.model_type == 'FFNet':
            self.scaler = StandardScaler()

    def create_encoded_features(self, features):
        if self.model_type == 'FFNet':
            return self.create_encoded_features_FFNet(features)
        else:
            raise Exception('Model type not supported')
    
    def create_encoded_features_FFNet(self, features):
        # Transform features into a pandas dataframe
        df = pd.DataFrame(features, index=[0])
        # Encode features
        df = df.replace({True: 1, False: 0})
        df = df.replace('10.000.000.000', 1.0)

        # Scale the features
        df = self.scale_features(df)
        # Convert to tensor
        tensor = torch.FloatTensor(df.values)

        return tensor
    
    def scale_features(self, df):
        # Scale the features
        df = self.scaler.fit_transform(df)
        return df

    def remove_features(self, df):
        return df[self.model_features]

model_features = [
    "SubdomainLevel",
    "UrlLength",
    "NumDashInHostname",
    "TildeSymbol",
    "NumPercent",
    "NumAmpersand",
    "NumNumericChars",
    "DomainInSubdomains",
    "HttpsInHostname",
    "PathLength",
    "DoubleSlashInPath",
    "PctExtResourceUrls",
    "InsecureForms",
    "ExtFormAction",
    "PopUpWindow",
    "IframeOrFrame",
    "ImagesOnlyInForm",
]
preprocessor = Preprocessor(model_features=model_features, model_type='FFNet')