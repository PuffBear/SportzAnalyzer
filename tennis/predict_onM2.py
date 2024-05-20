import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and feature columns
classifier = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Sample match data
matches = [
    ('Sebastian Baez', 'Alexander Zverev'),
    ('Ben Shelton', 'Harold Mayot'),
    ('Daniil Medvedev', 'Sebastian Baez'),
    ('Alexander Zverev', 'Daniil Medvedev')
]

# Load player statistics
player_df = pd.read_csv('CleanedMergedStatisticsLeaders.csv')

# Initialize the data for the new DataFrame
data = []

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
        'name': player1,
        'opponent_name': player2,
        '1st_Serve': p1_stats['1st_Serve'],
        '1st_Serve_won': p1_stats['1st_Serve_won'],
        '2nd_Serve_won': p1_stats['2nd_Serve_won'],
        '1st_Serve_Return_won': p1_stats['1st_Serve_Return_won'],
        '2nd_Serve_Return_won': p1_stats['2nd_Serve_Return_won'],
        'Return_points_won': p1_stats['Return_points_won']
    }
    
    # Append the row to the data list
    data.append(row)

# Create the new DataFrame with the specified columns
df_new = pd.DataFrame(data)

# Display the new DataFrame
print(df_new)

# Convert percentage strings to numerical values
percentage_columns = ['1st_Serve', '1st_Serve_won', '2nd_Serve_won', '1st_Serve_Return_won', '2nd_Serve_Return_won', 'Return_points_won']

for col in percentage_columns:
    # Convert to string and then strip '%'
    df_new[col] = df_new[col].astype(str).str.rstrip('%').astype(float)

# Get dummy variables for categorical features
df_new_encoded = pd.get_dummies(df_new)

# Ensure the new data has the same features as the training data
df_new_encoded = df_new_encoded.reindex(columns=feature_columns, fill_value=0)

# Normalize the new data using the same scaler
X_new_scaled = scaler.transform(df_new_encoded)

# Predict the probabilities of each player winning
probabilities = classifier.predict_proba(X_new_scaled)

# Display the probabilities
for i, row in df_new.iterrows():
    player1 = row['name']
    player2 = row['opponent_name']
    print(f"Match {i+1}:")
    print(f"  Probability of {player1} winning: {probabilities[i][1]:.2f}")
    print(f"  Probability of {player2} winning: {probabilities[i][0]:.2f}")
