import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class CarPricePredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_columns = None
        
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(['selling_price'], axis=1)
        y = df['selling_price']
        
        # Log transform target for better performance
        y = np.log1p(y)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        # Convert to numpy arrays if they are DataFrames
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # Linear Regression
        print("Training Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['linear_regression'] = lr
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            # Convert to numpy arrays if needed
            if hasattr(X_test, 'values'):
                X_test_np = X_test.values
            else:
                X_test_np = X_test
                
            if hasattr(y_test, 'values'):
                y_test_np = y_test.values
            else:
                y_test_np = y_test
            
            y_pred = model.predict(X_test_np)
            
            # Convert back from log scale
            y_test_original = np.expm1(y_test_np)
            y_pred_original = np.expm1(y_pred)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_original, y_pred_original)
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred_original)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"MAE: ₹{mae:,.2f}")
            print(f"RMSE: ₹{rmse:,.2f}")
            print(f"R² Score: {r2:.4f}")
        
        return results
    
    def predict_price(self, input_data, model_name='random_forest'):
        """Predict price for single input"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input
        processed_input = self.preprocessor.preprocess(input_df, training=False)
        
        # Ensure all columns are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in processed_input.columns:
                    processed_input[col] = 0
            
            # Reorder columns
            processed_input = processed_input[self.feature_columns]
        
        # Convert to numpy array for prediction
        if hasattr(processed_input, 'values'):
            processed_input_np = processed_input.values
        else:
            processed_input_np = processed_input
        
        # Make prediction
        model = self.models[model_name]
        prediction_log = model.predict(processed_input_np)[0]
        
        # Convert back from log scale
        prediction = np.expm1(prediction_log)
        
        return float(prediction)
    
    def save_models(self, path='models/'):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'{path}/{name}_model.pkl')
        
        # Save feature columns
        if self.feature_columns:
            joblib.dump(self.feature_columns, f'{path}/feature_columns.pkl')
    
    def load_models(self, path='models/'):
        """Load trained models"""
        self.models = {}
        
        model_files = {
            'linear_regression': f'{path}/linear_regression_model.pkl',
            'random_forest': f'{path}/random_forest_model.pkl'
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
        
        # Load feature columns
        if os.path.exists(f'{path}/feature_columns.pkl'):
            self.feature_columns = joblib.load(f'{path}/feature_columns.pkl')