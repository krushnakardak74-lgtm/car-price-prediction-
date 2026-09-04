import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UsedCarsDatasetGenerator:
    def __init__(self, seed=42):
        self.fake = Faker()
        np.random.seed(seed)
        random.seed(seed)
        
        # Define car brands and models
        self.brand_models = {
            'Maruti Suzuki': ['Swift', 'Dzire', 'Baleno', 'Wagon R', 'Alto', 'Ertiga', 'Vitara Brezza', 'Ciaz'],
            'Hyundai': ['i20', 'i10', 'Creta', 'Verna', 'Santro', 'Venue', 'Aura', 'Elantra'],
            'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'Civic', 'CR-V'],
            'Toyota': ['Innova', 'Fortuner', 'Etios', 'Glanza', 'Urban Cruiser', 'Camry'],
            'Mahindra': ['Scorpio', 'XUV500', 'Bolero', 'Thar', 'XUV300', 'Marazzo'],
            'Tata': ['Nexon', 'Tiago', 'Harrier', 'Altroz', 'Safari', 'Tigor', 'Punch'],
            'Renault': ['Duster', 'Kwid', 'Triber', 'Kiger'],
            'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire'],
            'Kia': ['Seltos', 'Sonet', 'Carnival'],
            'MG': ['Hector', 'Astor', 'Gloster'],
            'Skoda': ['Rapid', 'Kushaq', 'Slavia'],
            'Volkswagen': ['Polo', 'Vento', 'Taigun', 'Virtus'],
            'BMW': ['3 Series', '5 Series', 'X1', 'X3', 'X5'],
            'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLA', 'GLC'],
            'Audi': ['A4', 'A6', 'Q3', 'Q5', 'Q7']
        }
        
        # Define fuel types with probabilities
        self.fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
        self.fuel_probs = [0.6, 0.3, 0.05, 0.03, 0.02]
        
        # Seller types
        self.seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
        self.seller_probs = [0.5, 0.3, 0.2]
        
        # Transmission types
        self.transmissions = ['Manual', 'Automatic']
        self.transmission_probs = [0.7, 0.3]
        
        # Owner types
        self.owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
        self.owner_probs = [0.4, 0.35, 0.2, 0.05]
        
        # Seats options
        self.seats_options = [2, 4, 5, 6, 7, 8]
        self.seats_probs = [0.02, 0.05, 0.7, 0.1, 0.1, 0.03]
        
    def generate_car_name(self, brand, model):
        """Generate car name with variant"""
        variants = ['VXI', 'ZXI', 'VDI', 'ZDI', 'E', 'EX', 'S', 'S+', 'Limited Edition', 'Premium']
        variant = random.choice(variants)
        year = random.randint(2015, 2023)
        return f"{brand} {model} {variant} {year}"
    
    def generate_year(self):
        """Generate manufacturing year"""
        # More recent years are more common
        years = list(range(2000, 2024))
        weights = [np.exp(-0.1 * (2023 - year)) for year in years]
        weights = [w/sum(weights) for w in weights]
        return np.random.choice(years, p=weights)
    
    def generate_km_driven(self, year):
        """Generate kilometers driven based on car age"""
        current_year = 2024
        car_age = current_year - year
        
        # Average km per year
        avg_km_per_year = random.randint(10000, 20000)
        base_km = car_age * avg_km_per_year
        
        # Add some variation
        variation = np.random.normal(0, 0.2)
        km_driven = int(base_km * (1 + variation))
        
        # Ensure positive and reasonable
        km_driven = max(1000, min(km_driven, 300000))
        
        return km_driven
    
    def generate_engine_capacity(self, brand):
        """Generate engine capacity based on brand segment"""
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi']
        suv_brands = ['Mahindra', 'Toyota', 'Ford', 'MG']
        
        if brand in luxury_brands:
            return random.choice([1998, 1995, 2993, 2998, 1984])
        elif brand in suv_brands:
            return random.choice([1498, 1999, 2179, 2477, 2694])
        else:
            return random.choice([796, 998, 1197, 1199, 1248, 1462, 1498])
    
    def generate_max_power(self, engine_capacity, fuel_type):
        """Generate max power based on engine capacity and fuel type"""
        # Power per 100cc for different fuel types
        if fuel_type == 'Diesel':
            power_per_100cc = random.uniform(4.5, 6.0)
        elif fuel_type == 'Petrol':
            power_per_100cc = random.uniform(6.0, 8.0)
        elif fuel_type == 'Electric':
            power_per_100cc = random.uniform(8.0, 12.0)
        else:
            power_per_100cc = random.uniform(5.0, 6.5)
        
        base_power = (engine_capacity / 100) * power_per_100cc
        
        # Add some variation
        variation = np.random.normal(0, 0.1)
        max_power = base_power * (1 + variation)
        
        return round(max_power, 1)
    
    def generate_mileage(self, fuel_type, engine_capacity):
        """Generate mileage based on fuel type and engine"""
        if fuel_type == 'Petrol':
            base_mileage = random.uniform(15, 25)
        elif fuel_type == 'Diesel':
            base_mileage = random.uniform(18, 28)
        elif fuel_type == 'CNG':
            base_mileage = random.uniform(20, 30)
        elif fuel_type == 'Electric':
            base_mileage = random.uniform(100, 200)  # km per charge
        else:
            base_mileage = random.uniform(12, 20)
        
        # Larger engines typically have lower mileage
        engine_factor = 1 - ((engine_capacity - 1000) / 10000)
        mileage = base_mileage * engine_factor
        
        return round(mileage, 1)
    
    def generate_seats(self, brand):
        """Generate number of seats based on brand"""
        suv_brands = ['Mahindra', 'Toyota', 'Ford', 'MG', 'Tata']
        if brand in suv_brands:
            return random.choices(self.seats_options, weights=[0, 0.1, 0.4, 0.3, 0.15, 0.05])[0]
        else:
            return random.choices(self.seats_options, weights=[0.05, 0.1, 0.7, 0.1, 0.05, 0])[0]
    
    def calculate_base_price(self, brand, model, year, fuel_type, transmission):
        """Calculate base price for the car"""
        # Base prices for brands (in lakhs)
        brand_base_price = {
            'Maruti Suzuki': 3.5, 'Hyundai': 4.0, 'Honda': 6.0, 'Toyota': 8.0,
            'Mahindra': 7.0, 'Tata': 4.5, 'Renault': 4.0, 'Ford': 6.5,
            'Kia': 7.0, 'MG': 12.0, 'Skoda': 9.0, 'Volkswagen': 8.0,
            'BMW': 35.0, 'Mercedes-Benz': 40.0, 'Audi': 38.0
        }
        
        base_price = brand_base_price.get(brand, 5.0)  # in lakhs
        
        # Year depreciation
        current_year = 2024
        age = current_year - year
        depreciation_factor = (0.85 ** age)  # 15% depreciation per year
        
        # Fuel type factor
        fuel_factor = 1.0
        if fuel_type == 'Diesel':
            fuel_factor = 1.1
        elif fuel_type == 'Electric':
            fuel_factor = 1.3
        elif fuel_type in ['CNG', 'LPG']:
            fuel_factor = 0.9
        
        # Transmission factor
        transmission_factor = 1.2 if transmission == 'Automatic' else 1.0
        
        # Calculate final price in lakhs
        price_lakhs = base_price * depreciation_factor * fuel_factor * transmission_factor
        
        # Convert to rupees
        price_rupees = price_lakhs * 100000
        
        return price_rupees
    
    def apply_price_variation(self, base_price, km_driven, owner_type, seller_type):
        """Apply variations based on other factors"""
        # Kilometer effect
        km_factor = 1 - (km_driven / 2000000)  # Max effect at 200,000 km
        
        # Owner effect
        owner_multiplier = {
            'First Owner': 1.0,
            'Second Owner': 0.9,
            'Third Owner': 0.8,
            'Fourth & Above Owner': 0.7
        }
        
        # Seller effect
        seller_multiplier = {
            'Individual': 0.95,
            'Dealer': 1.0,
            'Trustmark Dealer': 1.1
        }
        
        # Apply all factors
        adjusted_price = base_price * km_factor * owner_multiplier[owner_type] * seller_multiplier[seller_type]
        
        # Add random variation
        random_variation = np.random.normal(1, 0.1)
        final_price = adjusted_price * random_variation
        
        # Ensure minimum price
        final_price = max(50000, final_price)
        
        return round(final_price)
    
    def generate_torque(self, engine_capacity, fuel_type):
        """Generate torque value"""
        if fuel_type == 'Diesel':
            torque_per_100cc = random.uniform(14, 18)
        elif fuel_type == 'Petrol':
            torque_per_100cc = random.uniform(9, 13)
        else:
            torque_per_100cc = random.uniform(10, 15)
        
        torque = (engine_capacity / 100) * torque_per_100cc
        return f"{round(torque)} Nm"
    
    def generate_single_car(self):
        """Generate a single car record"""
        # Select random brand and model
        brand = random.choice(list(self.brand_models.keys()))
        model = random.choice(self.brand_models[brand])
        
        # Generate car name
        name = self.generate_car_name(brand, model)
        
        # Generate other attributes
        year = self.generate_year()
        km_driven = self.generate_km_driven(year)
        fuel = random.choices(self.fuel_types, weights=self.fuel_probs)[0]
        seller_type = random.choices(self.seller_types, weights=self.seller_probs)[0]
        transmission = random.choices(self.transmissions, weights=self.transmission_probs)[0]
        owner = random.choices(self.owner_types, weights=self.owner_probs)[0]
        
        # Generate technical specifications
        engine = self.generate_engine_capacity(brand)
        max_power = self.generate_max_power(engine, fuel)
        mileage = self.generate_mileage(fuel, engine)
        seats = self.generate_seats(brand)
        torque = self.generate_torque(engine, fuel)
        
        # Calculate selling price
        base_price = self.calculate_base_price(brand, model, year, fuel, transmission)
        selling_price = self.apply_price_variation(base_price, km_driven, owner, seller_type)
        
        return {
            'name': name,
            'year': year,
            'selling_price': selling_price,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'torque': torque,
            'seats': seats
        }
    
    def generate_dataset(self, n_samples=10000):
        """Generate the complete dataset"""
        print(f"Generating {n_samples} car records...")
        
        records = []
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} records...")
            records.append(self.generate_single_car())
        
        df = pd.DataFrame(records)
        
        # Add some missing values realistically (5% of data)
        for col in ['mileage', 'engine', 'max_power', 'seats']:
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan
        
        # Add some outliers (1% of data)
        n_outliers = int(0.01 * len(df))
        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
        
        for idx in outlier_indices:
            # Make some prices too high or too low
            if np.random.random() > 0.5:
                df.loc[idx, 'selling_price'] *= np.random.uniform(3, 5)
            else:
                df.loc[idx, 'selling_price'] *= np.random.uniform(0.1, 0.3)
            
            # Make some km_driven unrealistic
            if np.random.random() > 0.7:
                df.loc[idx, 'km_driven'] *= np.random.uniform(3, 5)
        
        print("Dataset generation complete!")
        return df
    
    def save_dataset(self, df, filepath='data/used_cars.csv'):
        """Save dataset to CSV"""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        
    def analyze_dataset(self, df):
        """Print dataset analysis"""
        print("\n=== DATASET ANALYSIS ===")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        print("\n=== PRICE DISTRIBUTION ===")
        print(f"Min price: ₹{df['selling_price'].min():,.0f}")
        print(f"Max price: ₹{df['selling_price'].max():,.0f}")
        print(f"Mean price: ₹{df['selling_price'].mean():,.0f}")
        print(f"Median price: ₹{df['selling_price'].median():,.0f}")
        
        print("\n=== FUEL TYPE DISTRIBUTION ===")
        print(df['fuel'].value_counts())
        
        print("\n=== BRAND DISTRIBUTION (Top 10) ===")
        # Extract brand from name
        df['brand'] = df['name'].str.split().str[0]
        print(df['brand'].value_counts().head(10))
        
        print("\n=== MISSING VALUES ===")
        print(df.isnull().sum())
        
        print("\n=== DATA TYPES ===")
        print(df.dtypes)

def main():
    """Main function to generate and save dataset"""
    generator = UsedCarsDatasetGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_dataset(n_samples=10000)
    
    # Analyze dataset
    generator.analyze_dataset(df)
    
    # Save dataset
    generator.save_dataset(df, 'data/used_cars.csv')
    
    # Create a smaller sample for testing
    sample_df = df.sample(1000, random_state=42)
    generator.save_dataset(sample_df, 'data/used_cars_sample.csv')
    
    print("\n=== SAMPLE DATA (First 5 rows) ===")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()