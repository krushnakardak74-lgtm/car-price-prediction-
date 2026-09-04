import pandas as pd
import numpy as np

def create_simple_dataset(n_samples=5000):
    """Create a simple used cars dataset"""
    
    np.random.seed(42)
    
    # Brands and their price ranges (in lakhs)
    brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Tata', 'Mahindra', 'Renault', 'Kia', 'MG']
    base_prices = [3, 4, 6, 8, 7, 5, 7, 4, 7, 12]  # in lakhs
    
    # Models for each brand
    models = {
        'Maruti': ['Swift', 'Dzire', 'Baleno', 'Wagon R'],
        'Hyundai': ['i20', 'i10', 'Creta', 'Verna'],
        'Honda': ['City', 'Amaze', 'Jazz'],
        'Toyota': ['Innova', 'Fortuner', 'Etios'],
        'Ford': ['EcoSport', 'Endeavour', 'Figo'],
        'Tata': ['Nexon', 'Tiago', 'Harrier'],
        'Mahindra': ['Scorpio', 'XUV500', 'Bolero'],
        'Renault': ['Duster', 'Kwid'],
        'Kia': ['Seltos', 'Sonet'],
        'MG': ['Hector', 'Astor']
    }
    
    data = []
    
    for i in range(n_samples):
        # Select random brand
        brand_idx = np.random.randint(0, len(brands))
        brand = brands[brand_idx]
        model = np.random.choice(models[brand])
        
        # Generate year (2010-2023)
        year = np.random.randint(2010, 2024)
        
        # Generate km_driven based on year
        age = 2024 - year
        km_driven = np.random.randint(age * 5000, age * 25000)
        
        # Fuel type
        fuel = np.random.choice(['Petrol', 'Diesel', 'CNG'], p=[0.6, 0.35, 0.05])
        
        # Transmission
        transmission = np.random.choice(['Manual', 'Automatic'], p=[0.7, 0.3])
        
        # Seller type
        seller_type = np.random.choice(['Individual', 'Dealer', 'Trustmark Dealer'], 
                                       p=[0.5, 0.3, 0.2])
        
        # Owner
        owner = np.random.choice(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'],
                                 p=[0.4, 0.35, 0.2, 0.05])
        
        # Engine capacity (CC)
        if brand in ['Toyota', 'Ford', 'Mahindra', 'MG']:
            engine = np.random.choice([1498, 1999, 2179, 2477])
        else:
            engine = np.random.choice([796, 998, 1197, 1248, 1462])
        
        # Max power (bhp)
        if fuel == 'Diesel':
            max_power = engine * np.random.uniform(0.045, 0.06) / 100
        else:
            max_power = engine * np.random.uniform(0.06, 0.08) / 100
        
        # Seats
        if brand in ['Toyota', 'Mahindra']:
            seats = np.random.choice([5, 6, 7, 8], p=[0.3, 0.3, 0.3, 0.1])
        else:
            seats = np.random.choice([4, 5, 6], p=[0.1, 0.8, 0.1])
        
        # Calculate selling price
        base_price_lakhs = base_prices[brand_idx]
        
        # Depreciation based on year
        depreciation = 0.85 ** (2024 - year)
        
        # Fuel type adjustment
        if fuel == 'Diesel':
            fuel_factor = 1.1
        elif fuel == 'CNG':
            fuel_factor = 0.9
        else:
            fuel_factor = 1.0
        
        # Transmission adjustment
        if transmission == 'Automatic':
            trans_factor = 1.2
        else:
            trans_factor = 1.0
        
        # Owner adjustment
        owner_factor = {'First Owner': 1.0, 'Second Owner': 0.9, 
                       'Third Owner': 0.8, 'Fourth & Above Owner': 0.7}[owner]
        
        # Kilometer adjustment
        km_factor = 1 - (km_driven / 500000)
        
        # Calculate final price in rupees
        price_lakhs = (base_price_lakhs * depreciation * fuel_factor * 
                      trans_factor * owner_factor * km_factor)
        selling_price = price_lakhs * 100000 * np.random.uniform(0.9, 1.1)
        
        # Add some noise
        selling_price = selling_price * np.random.uniform(0.95, 1.05)
        
        # Create car name
        name = f"{brand} {model} {year}"
        
        # Create record
        record = {
            'name': name,
            'year': year,
            'selling_price': int(selling_price),
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': np.random.uniform(15, 25),
            'engine': engine,
            'max_power': round(max_power, 1),
            'torque': f"{int(engine * np.random.uniform(0.1, 0.15))} Nm",
            'seats': int(seats)
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add 5% missing values
    for col in ['mileage', 'engine', 'max_power', 'seats']:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    return df

# Generate and save dataset
df = create_simple_dataset(5000)
df.to_csv('data/used_cars_simple.csv', index=False)

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nPrice statistics:")
print(f"Min: ₹{df['selling_price'].min():,.0f}")
print(f"Max: ₹{df['selling_price'].max():,.0f}")
print(f"Mean: ₹{df['selling_price'].mean():,.0f}")
print(f"Median: ₹{df['selling_price'].median():,.0f}")