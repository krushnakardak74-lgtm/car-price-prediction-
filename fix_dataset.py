import pandas as pd
import numpy as np
import os

def fix_dataset():
    """Fix the existing dataset to ensure proper formatting"""
    
    # Read the existing dataset
    df = pd.read_csv('data/used_cars.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Ensure all required columns are present
    required_columns = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 
                       'seller_type', 'transmission', 'owner', 'mileage', 
                       'engine', 'max_power', 'torque', 'seats']
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return
    
    # Clean the data
    # 1. Ensure numeric columns are numeric
    numeric_columns = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Remove rows with missing essential values
    essential_cols = ['year', 'selling_price', 'km_driven', 'fuel', 'engine']
    df = df.dropna(subset=essential_cols)
    
    # 3. Remove unrealistic values
    df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
    df = df[(df['selling_price'] > 0) & (df['selling_price'] <= 50000000)]  # Max 5 crore
    df = df[(df['km_driven'] >= 0) & (df['km_driven'] <= 1000000)]  # Max 10 lakh km
    df = df[(df['engine'] >= 500) & (df['engine'] <= 5000)]  # Reasonable engine size
    
    # 4. Fill other missing values
    df['mileage'] = df['mileage'].fillna(df['mileage'].median() if not df['mileage'].empty else 18)
    df['max_power'] = df['max_power'].fillna(df['max_power'].median() if not df['max_power'].empty else 100)
    df['seats'] = df['seats'].fillna(5)
    
    # 5. Ensure categorical columns have proper values
    categorical_mapping = {
        'fuel': ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],
        'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer'],
        'transmission': ['Manual', 'Automatic'],
        'owner': ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
    }
    
    for col, valid_values in categorical_mapping.items():
        if col in df.columns:
            # Replace invalid values with the most common valid value
            mask = ~df[col].isin(valid_values)
            if mask.any():
                most_common = df[col].mode()[0] if not df[col].mode().empty else valid_values[0]
                df.loc[mask, col] = most_common
    
    # 6. Remove torque_numeric column if it exists (it will be recreated)
    if 'torque_numeric' in df.columns:
        df = df.drop('torque_numeric', axis=1)
    
    # Save the cleaned dataset
    df.to_csv('data/used_cars_cleaned.csv', index=False)
    
    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Rows removed: {100 - len(df)/100*100:.1f}%")
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Also create a backup of the original
    os.makedirs('data/backup', exist_ok=True)
    pd.read_csv('data/used_cars.csv').to_csv('data/backup/used_cars_original.csv', index=False)
    
    # Replace the original with cleaned version
    df.to_csv('data/used_cars.csv', index=False)
    
    print(f"\nDataset fixed and saved to data/used_cars.csv")
    print(f"Original backed up to data/backup/used_cars_original.csv")
    
    return df

if __name__ == "__main__":
    fix_dataset()