# generate_csv.py
import numpy as np
import pandas as pd

np.random.seed(101)
n_rows = 200

# Realistic transaction data for batch processing
accounts = [f"ACC-{np.random.randint(10000, 99999)}" for _ in range(n_rows)]
amounts = np.round(np.random.exponential(scale=65, size=n_rows) + 10, 2)
hours = np.random.randint(0, 24, size=n_rows)
distances = np.round(np.random.exponential(scale=12, size=n_rows) + 1, 1)
velocities = np.random.poisson(lam=3, size=n_rows)

# Injecting some clear fraud outliers
amounts[12] = 1450.00
hours[12] = 3
distances[12] = 230.5
velocities[12] = 14

amounts[45] = 980.50
hours[45] = 2
distances[45] = 185.0
velocities[45] = 11

df = pd.DataFrame({
    "account_id": accounts,
    "amount": amounts,
    "hour_of_day": hours,
    "distance_from_home_km": distances,
    "velocity_last_24h": velocities,
})

df.to_csv("transactions.csv", index=False)
print("transactions.csv generated with 200 records!")