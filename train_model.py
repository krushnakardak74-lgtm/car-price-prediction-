import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import CarDataPreprocessor
from ml_model import CarPricePredictor
import joblib
import os
import sys

def main():
    try:
        # Initialize components
        preprocessor = CarDataPreprocessor()
        predictor = CarPricePredictor()
        
        # Define dataset paths to try
        dataset_paths = [
            'data/used_cars.csv',
            'data/used_cars_simple.csv',
            'data/working_cars.csv',
            'data/test_cars.csv'
        ]
        
        # Load and preprocess data
        print("Loading data...")
        
        df = None
        for path in dataset_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                df = preprocessor.load_data(path)
                break
        
        if df is None:
            print("No dataset found. Creating a sample dataset...")
            # Try to import and run test_dataset
            try:
                from test_dataset import create_test_dataset
                create_test_dataset()
                df = preprocessor.load_data('data/test_cars.csv')
            except ImportError:
                # Create a minimal dataset directly
                print("Creating minimal dataset directly...")
                import numpy as np
                
                # Create a very simple dataset
                data = {
                    'name': ['Maruti Swift 2020', 'Hyundai i20 2019', 'Honda City 2018'],
                    'year': [2020, 2019, 2018],
                    'selling_price': [600000, 550000, 800000],
                    'km_driven': [40000, 50000, 60000],
                    'fuel': ['Petrol', 'Petrol', 'Diesel'],
                    'seller_type': ['Individual', 'Dealer', 'Individual'],
                    'transmission': ['Manual', 'Manual', 'Automatic'],
                    'owner': ['First Owner', 'First Owner', 'Second Owner'],
                    'mileage': [20.5, 19.2, 18.5],
                    'engine': [1200, 1200, 1500],
                    'max_power': [82.0, 80.5, 100.0],
                    'torque': ['113 Nm', '115 Nm', '200 Nm'],
                    'seats': [5, 5, 5]
                }
                
                df = pd.DataFrame(data)
                os.makedirs('data', exist_ok=True)
                df.to_csv('data/minimal_cars.csv', index=False)
                df = preprocessor.load_data('data/minimal_cars.csv')
        
        if df is None or df.empty:
            print("ERROR: Could not load or create dataset!")
            return
        
        print(f"Dataset loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("Preprocessing data...")
        df_processed = preprocessor.preprocess(df, training=True)
        
        # Check if we have the target column
        if 'selling_price' not in df_processed.columns:
            print("ERROR: 'selling_price' column not found in processed data!")
            print(f"Available columns: {list(df_processed.columns)}")
            return
        
        print(f"Processed dataset shape: {df_processed.shape}")
        print(f"Processed columns: {list(df_processed.columns)}")
        
        # Prepare data for training
        predictor.set_preprocessor(preprocessor)
        X, y = predictor.prepare_data(df_processed)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        if X.shape[0] == 0:
            print("ERROR: No data available for training!")
            return
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train models
        print("\nTraining models...")
        predictor.train_models(X_train, y_train)
        
        # Evaluate models
        print("\nEvaluating models...")
        results = predictor.evaluate_models(X_test, y_test)
        
        # Save everything
        print("\nSaving models and preprocessor...")
        os.makedirs('models', exist_ok=True)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/')
        
        # Save models
        predictor.save_models('models/')
        
        # Save metadata
        metadata = {
            'feature_columns': predictor.feature_columns,
            'model_performance': results
        }
        joblib.dump(metadata, 'models/metadata.pkl')
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        
        if 'random_forest' in results:
            print(f"Best model: Random Forest (R² = {results['random_forest']['R2']:.4f})")
            print(f"\nModel Performance:")
            print(f"MAE: ₹{results['random_forest']['MAE']:,.2f}")
            print(f"RMSE: ₹{results['random_forest']['RMSE']:,.2f}")
            print(f"R² Score: {results['random_forest']['R2']:.4f}")
        
        # Test prediction with sample data
        print("\n" + "="*50)
        print("Testing with sample prediction...")
        sample_data = {
            'year': 2019,
            'km_driven': 50000,
            'fuel': 'Petrol',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner',
            'engine': 1200,
            'max_power': 82,
            'seats': 5,
            'name': 'Maruti Suzuki Test'
        }
        
        try:
            prediction = predictor.predict_price(sample_data)
            print(f"Sample prediction: ₹{prediction:,.2f}")
        except Exception as e:
            print(f"Sample prediction failed: {e}")
            print("This is OK for the first run.")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if dataset exists in data/ folder")
        print("2. Verify dataset has required columns")
        print("3. Check for any data type issues")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()