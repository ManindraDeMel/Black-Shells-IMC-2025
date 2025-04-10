import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("Data/prices_round_1_day_-2.csv", sep=';')

# Convert 'timestamp' to numeric if not already
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Group by product
products = df['product'].unique()

# Plot mid prices for each product
plt.figure(figsize=(12, 6))
for product in products:
    product_df = df[df['product'] == product]
    plt.plot(product_df['timestamp'], product_df['mid_price'], label=product)

plt.title('Mid Prices Over Time for Each Product')
plt.xlabel('Timestamp')
plt.ylabel('Mid Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate and display standard deviation of mid prices for each product
print("Standard Deviations of Mid Prices by Product:")
for product in products:
    product_df = df[df['product'] == product]
    std_dev = product_df['mid_price'].std()
    print(f"{product}: {std_dev:.2f}")

# Plot histograms of bid_volume_1 and bid_volume_2 for each product
for product in products:
    product_df = df[df['product'] == product]
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(product_df['bid_volume_1'].dropna(), bins=15, color='skyblue', edgecolor='black')
    plt.title(f'{product} - Bid Volume 1')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(product_df['bid_volume_2'].dropna(), bins=15, color='salmon', edgecolor='black')
    plt.title(f'{product} - Bid Volume 2')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
