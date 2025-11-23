import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_retail_data(num_days=365, num_stores=10, num_products=50):
    """Generate realistic retail sales data"""
    
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(num_days)]
    
    store_ids = [f"ST{8000 + i}" for i in range(num_stores)]
    sku_ids = [f"SKU{20000 + i}" for i in range(num_products)]
    
    data = []
    record_id = 1
    
    for date in dates:
        is_weekend = date.weekday() >= 5
        weekend_multiplier = 1.3 if is_weekend else 1.0
        
        is_holiday = (date.month == 12 and date.day >= 20) or (date.month == 1 and date.day <= 5)
        holiday_multiplier = 1.5 if is_holiday else 1.0
        
        seasonal_multiplier = 1.2 if date.month in [11, 12] else 0.9 if date.month in [1, 2] else 1.0
        
        for store_id in store_ids:
            store_performance = random.uniform(0.7, 1.3)
            
            products_sold = random.randint(5, len(sku_ids)//2)
            selected_products = random.sample(sku_ids, products_sold)
            
            for sku_id in selected_products:
                base_sales = random.randint(10, 100)
                
                final_sales = int(base_sales * weekend_multiplier * holiday_multiplier * seasonal_multiplier * store_performance * random.uniform(0.8, 1.2))
                
                base_price = random.uniform(50, 1000)
                is_featured = random.choice([0, 0, 0, 1])
                is_display = random.choice([0, 0, 0, 1])
                
                if is_featured:
                    total_price = base_price * random.uniform(0.8, 0.95)
                else:
                    total_price = base_price * random.uniform(0.95, 1.05)
                    
                data.append({
                    'record_ID': record_id,
                    'week': date.strftime('%d-%m-%Y'),
                    'store_id': store_id,
                    'sku_id': sku_id, 
                    'total_price': round(total_price, 2),
                    'base_price': round(base_price, 2),
                    'is_featured_sku': is_featured,
                    'is_display_sku': is_display,
                    'units_sold': max(1, final_sales),
                })
                
                record_id += 1
                
    return pd.DataFrame(data)

# Generate different sized datasets
def create_test_datasets():
    """Create multiple test datasets of different sizes"""
    
    datasets = {
        'child_retail_data.csv': generate_retail_data(num_days=90, num_stores=15, num_products=20),
        'teen_retail_data.csv': generate_retail_data(num_days=180, num_stores=20, num_products=30),
        'adult_retail_data.csv': generate_retail_data(num_days=365, num_stores=25, num_products=50),
        'boomer_retail_data.csv': generate_retail_data(num_days=730, num_stores=35, num_products=75),
        'daddy_retail_data.csv': generate_retail_data(num_days=1095, num_stores=50, num_products=105)
    }
    
    for filename, df in datasets.items():
        df.to_csv(filename, index=False)
        print(f"✅ Generated {filename}: {len(df):,} records")
        print(f"   📅 Date range: {df['week'].min()} to {df['week'].max()}")
        print(f"   🏪 Stores: {df['store_id'].nunique()}")
        print(f"   📦 Products: {df['sku_id'].nunique()}") 
        print(f"   💰 Total sales: {df['units_sold'].sum():,} units")
        print()

if __name__ == "__main__":
    create_test_datasets()
    print("🎉 All test datasets generated successfully!")