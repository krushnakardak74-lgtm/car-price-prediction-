import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class CarDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_features = ['year', 'km_driven', 'engine', 'max_power', 'seats', 'car_age', 'power_to_weight']
        self.categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
        
    def load_data(self, filepath='data/used_cars.csv'):
        """Load and clean the dataset"""
        df = pd.read_csv(filepath)
        
        # Initial cleaning
        df = self._clean_data(df)
        return df
    
    def _clean_data(self, df):
        """Clean the raw data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle string columns that might be numeric
        self._convert_string_columns(df)
        
        # Fill missing values for numerical columns
        numerical_cols = ['engine', 'max_power', 'seats', 'mileage', 'year', 'km_driven', 'selling_price']
        for col in numerical_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if col != 'selling_price':  # Don't fill selling_price with median
                    df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
        
        # Handle selling_price column - ensure it's numeric
        if 'selling_price' in df.columns:
            df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
            
            # Remove rows with missing or negative prices
            df = df[df['selling_price'].notna()]
            df = df[df['selling_price'] > 0]
            
            # Remove outliers in selling_price using IQR
            Q1 = df['selling_price'].quantile(0.25)
            Q3 = df['selling_price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['selling_price'] >= lower_bound) & 
                   (df['selling_price'] <= upper_bound)]
        
        # Handle km_driven - ensure it's numeric and reasonable
        if 'km_driven' in df.columns:
            df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
            df = df[(df['km_driven'] > 0) & (df['km_driven'] <= 1000000)]
        
        # Handle year - ensure it's reasonable
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
        
        return df
    
    def _convert_string_columns(self, df):
        """Convert string columns to numeric where appropriate"""
        # Process engine column
        if 'engine' in df.columns and df['engine'].dtype == 'object':
            df['engine'] = df['engine'].astype(str).str.replace(' CC', '', regex=False)
            df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
        
        # Process max_power column
        if 'max_power' in df.columns and df['max_power'].dtype == 'object':
            df['max_power'] = df['max_power'].astype(str).str.replace(' bhp', '', regex=False)
            df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
        
        # Process mileage column
        if 'mileage' in df.columns and df['mileage'].dtype == 'object':
            df['mileage'] = df['mileage'].astype(str).str.replace(' kmpl', '', regex=False)
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        
        return df
    
    def preprocess(self, df, training=True):
        """Preprocess the data"""
        # Create features
        df = self._feature_engineering(df)
        
        # Handle categorical variables
        for feature in self.categorical_features:
            if feature in df.columns:
                # Handle NaN values in categorical columns
                df[feature] = df[feature].fillna('Unknown')
                
                if training:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
                else:
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        df[feature] = df[feature].apply(
                            lambda x: x if str(x) in self.label_encoders[feature].classes_ 
                            else 'Unknown'
                        )
                        df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
        
        # Ensure numerical features are numeric
        for feature in self.numerical_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                # Fill any remaining NaN values
                df[feature] = df[feature].fillna(df[feature].median() if not df[feature].empty else 0)
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        
        if training:
            if numerical_cols:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if numerical_cols:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def _feature_engineering(self, df):
        """Create new features"""
        # Car age
        current_year = 2024
        df['car_age'] = current_year - df['year']
        
        # Power to weight ratio (estimated)
        if 'max_power' in df.columns and 'engine' in df.columns:
            # Avoid division by zero
            df['power_to_weight'] = df['max_power'] / (df['engine'] * 0.1 + 1e-6)
        
        # Brand from name
        if 'name' in df.columns:
            df['brand'] = df['name'].str.split().str[0]
        
        # Drop unnecessary columns
        columns_to_drop = ['name', 'torque', 'mileage', 'torque_numeric']  # torque_numeric was created earlier
        for col in columns_to_drop:
            if col in df.columns and col not in ['selling_price'] + self.numerical_features + self.categorical_features:
                df = df.drop(col, axis=1, errors='ignore')
        
        return df
    
    def save_preprocessor(self, path='models/'):
        """Save preprocessing objects"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.label_encoders, f'{path}/label_encoders.pkl')
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.numerical_features, f'{path}/numerical_features.pkl')
        joblib.dump(self.categorical_features, f'{path}/categorical_features.pkl')
    
    def load_preprocessor(self, path='models/'):
        """Load preprocessing objects"""
        self.label_encoders = joblib.load(f'{path}/label_encoders.pkl')
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.numerical_features = joblib.load(f'{path}/numerical_features.pkl')
        self.categorical_features = joblib.load(f'{path}/categorical_features.pkl')