import pandas as pd
import numpy as np
import os

def create_working_dataset():
    """Create a clean, working dataset"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data
    data = {
        'name': [],
        'year': [],
        'selling_price': [],
        'km_driven': [],
        'fuel': [],
        'seller_type': [],
        'transmission': [],
        'owner': [],
        'mileage': [],
        'engine': [],
        'max_power': [],
        'torque': [],
        'seats': []
    }
    
    brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Tata', 'Mahindra']
    models = {
        'Maruti': ['Swift', 'Dzire', 'Baleno', 'Wagon R'],
        'Hyundai': ['i20', 'i10', 'Creta', 'Verna'],
        'Honda': ['City', 'Amaze', 'Jazz'],
        'Toyota': ['Innova', 'Fortuner', 'Etios'],
        'Ford': ['EcoSport', 'Endeavour', 'Figo'],
        'Tata': ['Nexon', 'Tiago', 'Harrier'],
        'Mahindra': ['Scorpio', 'XUV500', 'Bolero']
    }
    
    for i in range(n_samples):
        brand = np.random.choice(brands)
        model = np.random.choice(models[brand])
        year = np.random.randint(2010, 2024)
        
        # Generate other features
        km_driven = np.random.randint(10000, 150000)
        fuel = np.random.choice(['Petrol', 'Diesel'], p=[0.6, 0.4])
        seller_type = np.random.choice(['Individual', 'Dealer'], p=[0.6, 0.4])
        transmission = np.random.choice(['Manual', 'Automatic'], p=[0.7, 0.3])
        owner = np.random.choice(['First Owner', 'Second Owner', 'Third Owner'], 
                                p=[0.5, 0.3, 0.2])
        
        # Technical specs based on brand
        if brand in ['Toyota', 'Mahindra']:
            engine = np.random.choice([1498, 1999, 2179])
            seats = np.random.choice([5, 6, 7])
        else:
            engine = np.random.choice([796, 998, 1197, 1248])
            seats = 5
        
        max_power = engine * np.random.uniform(0.05, 0.07)
        mileage = np.random.uniform(15, 25)
        torque = f"{int(engine * np.random.uniform(0.1, 0.15))} Nm"
        
        # Calculate price
        base_price = {
            'Maruti': 500000, 'Hyundai': 600000, 'Honda': 800000,
            'Toyota': 1200000, 'Ford': 900000, 'Tata': 700000, 'Mahindra': 1100000
        }[brand]
        
        # Apply factors
        age_factor = (2024 - year) * 0.05  # 5% depreciation per year
        km_factor = (km_driven / 100000) * 0.1  # 10% per 100k km
        
        if fuel == 'Diesel':
            fuel_factor = 1.1
        else:
            fuel_factor = 1.0
            
        if transmission == 'Automatic':
            trans_factor = 1.2
        else:
            trans_factor = 1.0
            
        if owner == 'First Owner':
            owner_factor = 1.0
        elif owner == 'Second Owner':
            owner_factor = 0.9
        else:
            owner_factor = 0.8
        
        selling_price = base_price * (1 - age_factor) * (1 - km_factor) * fuel_factor * trans_factor * owner_factor
        selling_price *= np.random.uniform(0.9, 1.1)  # Random variation
        
        # Add to dataset
        data['name'].append(f"{brand} {model} {year}")
        data['year'].append(year)
        data['selling_price'].append(int(selling_price))
        data['km_driven'].append(km_driven)
        data['fuel'].append(fuel)
        data['seller_type'].append(seller_type)
        data['transmission'].append(transmission)
        data['owner'].append(owner)
        data['mileage'].append(round(mileage, 1))
        data['engine'].append(engine)
        data['max_power'].append(round(max_power, 1))
        data['torque'].append(torque)
        data['seats'].append(seats)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/working_cars.csv', index=False)
    
    print("Working dataset created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: data/working_cars.csv")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_working_dataset()