import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("atp_matches_2023.csv")

print(df.head())
print(df.tail())

print(df.columns)
df = df.dropna()

country_heights = df.groupby('player2_ioc')['player2_ht'].mean().reset_index()
country_heights.sort_values('player2_ht', inplace=True)  # Sorting helps in better visualization

# Reshape the data for heatmap (if necessary)
country_heights_pivot = country_heights.pivot("player2_ioc", "player2_ht", "player2_ht")


plt.figure(figsize=(12, 8))  # Set the figure size as needed
heatmap = sns.heatmap(country_heights_pivot, annot=True, cmap='viridis', fmt=".1f")
heatmap.set_title('Average Player2 Heights by Country')
plt.xlabel('Average Height')
plt.ylabel('Country')
plt.show()
