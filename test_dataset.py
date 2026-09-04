import pandas as pd
import numpy as np
import os

def create_test_dataset():
    """Create a small test dataset to verify the code works"""
    
    # Create a simple test dataset
    data = {
        'name': ['Maruti Suzuki Swift 2019', 'Hyundai i20 2020', 'Honda City 2018', 
                 'Toyota Fortuner 2017', 'Mahindra Scorpio 2019'],
        'year': [2019, 2020, 2018, 2017, 2019],
        'selling_price': [520000, 850000, 950000, 1800000, 1400000],
        'km_driven': [45000, 30000, 55000, 70000, 40000],
        'fuel': ['Petrol', 'Petrol', 'Petrol', 'Diesel', 'Diesel'],
        'seller_type': ['Individual', 'Dealer', 'Trustmark Dealer', 'Dealer', 'Individual'],
        'transmission': ['Manual', 'Manual', 'Automatic', 'Manual', 'Manual'],
        'owner': ['First Owner', 'First Owner', 'Second Owner', 'Second Owner', 'First Owner'],
        'mileage': [19.5, 18.2, 16.8, 12.5, 13.2],
        'engine': [1197, 1199, 1498, 2755, 2179],
        'max_power': [78.5, 82.3, 117.3, 174.5, 140.2],
        'torque': ['113 Nm', '115 Nm', '145 Nm', '450 Nm', '320 Nm'],
        'seats': [5, 5, 5, 7, 7]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/test_cars.csv', index=False)
    
    print("Test dataset created successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_test_dataset()