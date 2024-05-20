import pandas as pd
from model1 import classifier
from model1 import label_encoder
from sklearn.preprocessing import StandardScaler
from model1 import X_train

# Convert the sample data to a DataFrame
player_df = pd.read_csv('CleanedMergedStatisticsLeaders.csv')

# Initialize the data for the new DataFrame
data = []

# Example of how to map the provided player data to the new DataFrame
# Here, we're assuming you have match data that pairs player1 with player2
matches = [
    ('Sebastian Baez', 'Alexander Zverev'),  # Example match data: (player1, player2, winner)
    ('Ben Shelton', 'Harold Mayot'),
    ('Daniil Medvedev', 'Sebastian Baez'),
    ('Alexander Zverev', 'Daniil Medvedev')
]

for match in matches:
    player1 = match[0]
    player2 = match[1]
    
    # Get player1 stats
    p1_stats = player_df[player_df['name'] == player1]
    if p1_stats.empty:
        print(f"Player {player1} not found in the dataset.")
        continue  # Skip to the next match if player1 is not found

    # Get player2 stats
    p2_stats = player_df[player_df['name'] == player2]
    if p2_stats.empty:
        print(f"Player {player2} not found in the dataset.")
        continue  # Skip to the next match if player2 is not found

    # Extract the first row of stats for each player
    p1_stats = p1_stats.iloc[0]
    p2_stats = p2_stats.iloc[0]
    
    # Create a dictionary for the new DataFrame row
    row = {
        'player1_name': player1,
        'player2_name': player2,
        'p1_1st_Serve': p1_stats['1st_Serve'],
        'p1_1st_Serve_won': p1_stats['1st_Serve_won'],
        'p1_2nd_serve_won': p1_stats['2nd_Serve_won'],
        'p2_1st_Serve': p2_stats['1st_Serve'],
        'p2_1st_Serve_won': p2_stats['1st_Serve_won'],
        'p2_2nd_serve_won': p2_stats['2nd_Serve_won'],
    }
    
    # Append the row to the data list
    data.append(row)

# Create the new DataFrame with the specified columns
df_new = pd.DataFrame(data, columns=['player1_name', 'player2_name', 'p1_1st_Serve', 'p1_1st_Serve_won',
                                     'p1_2nd_serve_won', 'p2_1st_Serve',
                                     'p2_1st_Serve_won', 'p2_2nd_serve_won'])

# Display the new DataFrame
print(df_new)

# Convert percentage strings to numerical values
percentage_columns = ['p1_1st_Serve', 'p1_1st_Serve_won', 'p1_2nd_serve_won',
                      'p2_1st_Serve', 'p2_1st_Serve_won', 'p2_2nd_serve_won']

new_df = df_new

for col in percentage_columns:
    new_df[col] = new_df[col].str.rstrip('%').astype(float)

# Encode the player names using the same label encoder
new_df['player1_name'] = label_encoder.transform(new_df['player1_name'])
new_df['player2_name'] = label_encoder.transform(new_df['player2_name'])

# Ensure the new data has the same features as the training data
X_new = new_df

# Normalize the new data using the same scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on the training data
X_new_scaled = scaler.transform(X_new)

probabilities = classifier.predict_proba(X_new_scaled)

# Display the probabilities
for i, row in new_df.iterrows():
    player1 = label_encoder.inverse_transform([int(row['player1_name'])])[0]
    player2 = label_encoder.inverse_transform([int(row['player2_name'])])[0]
    print(f"Match {i+1}:")
    print(f"  Probability of {player1} winning: {probabilities[i][1]:.2f}")
    print(f"  Probability of {player2} winning: {probabilities[i][0]:.2f}")