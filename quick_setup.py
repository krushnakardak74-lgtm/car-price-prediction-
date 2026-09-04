#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

def setup_project():
    """Setup the project with all necessary files"""
    
    print("Setting up Car Price Prediction Project...")
    
    # Create directories
    directories = ['data', 'models', 'logs']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}/")
    
    # Create a working dataset
    print("\nCreating working dataset...")
    
    # Simple dataset
    data = {
        'name': [
            'Maruti Swift VXI 2020',
            'Hyundai i20 Sportz 2019', 
            'Honda City VX 2018',
            'Toyota Fortuner 4x2 2017',
            'Mahindra Scorpio S10 2019',
            'Tata Nexon XZ 2020',
            'Maruti Baleno Alpha 2021',
            'Hyundai Creta SX 2019',
            'Ford EcoSport Titanium 2018',
            'Renault Duster RXS 2016'
        ],
        'year': [2020, 2019, 2018, 2017, 2019, 2020, 2021, 2019, 2018, 2016],
        'selling_price': [650000, 580000, 850000, 1850000, 1450000, 1150000, 750000, 1250000, 780000, 620000],
        'km_driven': [35000, 42000, 52000, 68000, 38000, 22000, 15000, 32000, 58000, 82000],
        'fuel': ['Petrol', 'Petrol', 'Petrol', 'Diesel', 'Diesel', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
        'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer', 'Dealer', 'Individual', 'Trustmark Dealer', 'Individual', 'Dealer', 'Individual', 'Individual'],
        'transmission': ['Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Manual', 'Manual'],
        'owner': ['First Owner', 'First Owner', 'Second Owner', 'Second Owner', 'First Owner', 'First Owner', 'First Owner', 'First Owner', 'Second Owner', 'Third Owner'],
        'mileage': [21.5, 19.8, 17.2, 13.5, 14.2, 22.5, 20.8, 18.5, 16.2, 19.8],
        'engine': [1197, 1199, 1497, 2694, 2179, 1497, 1197, 1493, 1498, 1461],
        'max_power': [82.0, 81.0, 118.0, 175.0, 140.0, 108.5, 83.0, 113.0, 123.0, 108.5],
        'torque': ['113 Nm', '114 Nm', '145 Nm', '450 Nm', '320 Nm', '260 Nm', '113 Nm', '250 Nm', '149 Nm', '248 Nm'],
        'seats': [5, 5, 5, 7, 7, 5, 5, 5, 5, 5]
    }
    
    df = pd.DataFrame(data)
    
    # Create larger dataset by replicating with variations
    df_large = df.copy()
    for i in range(9):  # Create 100 samples total (10 * 10)
        df_copy = df.copy()
        # Add some variation
        df_copy['selling_price'] = df_copy['selling_price'] * np.random.uniform(0.9, 1.1, len(df_copy))
        df_copy['km_driven'] = (df_copy['km_driven'] * np.random.uniform(0.8, 1.2, len(df_copy))).astype(int)
        df_copy['mileage'] = df_copy['mileage'] * np.random.uniform(0.95, 1.05, len(df_copy))
        df_large = pd.concat([df_large, df_copy], ignore_index=True)
    
    # Save datasets
    df.to_csv('data/test_cars.csv', index=False)
    df_large.to_csv('data/used_cars.csv', index=False)
    
    print(f"Created test_cars.csv with {len(df)} samples")
    print(f"Created used_cars.csv with {len(df_large)} samples")
    
    # Create a simple version without string units
    df_simple = df_large.copy()
    # Remove units from torque for simplicity
    df_simple['torque'] = df_simple['torque'].str.replace(' Nm', '')
    df_simple.to_csv('data/used_cars_simple.csv', index=False)
    print(f"Created used_cars_simple.csv")
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nYou can now run:")
    print("1. python train_model.py  - to train the models")
    print("2. python app.py          - to start the API server")
    print("3. Open frontend/index.html in browser")
    
    return True

if __name__ == "__main__":
    setup_project()