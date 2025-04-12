import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the kelp data
kelp_data = pd.read_csv("round1db/prices_round_1_day_-2.csv", sep=";", header=0)

# Calculate mm_mid
def calculate_mm_mid(row):
    for i in range(1, 4):
        if row.get(f'bid_volume_{i}', 0) >= 20:
            best_bid = row.get(f'bid_price_{i}', None)
            break
    else:
        best_bid = None

    for i in range(1, 4):
        if row.get(f'ask_volume_{i}', 0) >= 20:
            best_ask = row.get(f'ask_price_{i}', None)
            break
    else:
        best_ask = None

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2
    return None

kelp_data['mm_mid'] = kelp_data.apply(calculate_mm_mid, axis=1)

# Create fair price DataFrame
kelp_fair_prices = kelp_data[['timestamp', 'mm_mid']].rename(columns={'mm_mid': 'fair'})

# Choose iteration count
iterations = 10

# Compute shifted prices and returns
kelp_fair_prices[f'fair_in_{iterations}_its'] = kelp_fair_prices['fair'].shift(-iterations)
kelp_fair_prices[f'fair_{iterations}_its_ago'] = kelp_fair_prices['fair'].shift(iterations)

kelp_fair_prices[f'returns_in_{iterations}_its'] = (
    kelp_fair_prices[f'fair_in_{iterations}_its'] - kelp_fair_prices['fair']
)
kelp_fair_prices[f'returns_from_{iterations}_its_ago'] = (
    kelp_fair_prices['fair'] - kelp_fair_prices[f'fair_{iterations}_its_ago']
)

# Drop NA values
kelp_fair_prices.dropna(inplace=True)

# Linear regression to estimate beta
X = kelp_fair_prices[[f'returns_from_{iterations}_its_ago']]
y = kelp_fair_prices[f'returns_in_{iterations}_its']
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

beta = model.coef_[0]
#beta
print(beta)