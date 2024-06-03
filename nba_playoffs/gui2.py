import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize the main window
root = tk.Tk()
root.title("Sports Analyzer")

# Load and prepare the data
df = pd.read_csv("nba_playoff_matchups.csv")
df = df.drop(columns=['GAME DATE', 'MIN'])
df = df.dropna()
label_encoder = LabelEncoder()
df['TEAM_1'] = label_encoder.fit_transform(df['TEAM_1'])
df['OPPONENT'] = label_encoder.fit_transform(df['OPPONENT'])
df['OUTCOME'] = label_encoder.fit_transform(df['OUTCOME'])

# Define features and target
X = df.drop(columns=['OUTCOME'])
y = df['OUTCOME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Function to handle analyze button click
def analyze_matchup():
    team1 = team1_entry.get()
    team2 = team2_entry.get()
    player_df = pd.read_csv('nba_playoff_team_data.csv')

    p1_stats = player_df[player_df['TEAM'] == team1]
    p2_stats = player_df[player_df['TEAM'] == team2]

    if p1_stats.empty or p2_stats.empty:
        messagebox.showerror("Error", "One or both teams not found in the dataset.")
        return

    p1_stats = p1_stats.iloc[0]
    p2_stats = p2_stats.iloc[0]

    row = {
        'TEAM_1': team1,
        'OPPONENT': team2,
        'PTS': p1_stats['PTS'],
        'FGM': p1_stats['FGM'],
        'FGA': p1_stats['FGA'],
        'FG%': p1_stats['FG%'],
        '3PM': p1_stats['3PM'],
        '3PA': p1_stats['3PA'],
        '3P%': p1_stats['3P%'],
        'FTM': p1_stats['FTM'],
        'FTA': p1_stats['FTA'],
        'FT%': p1_stats['FT%'],
        'OREB': p1_stats['OREB'],
        'DREB': p1_stats['DREB'],
        'REB': p1_stats['REB'],
        'AST': p1_stats['AST'],
        'STL': p1_stats['STL'],
        'BLK': p1_stats['BLK'],
        'TOV': p1_stats['TOV'],
        'PF': p1_stats['PF'],
        '+/-': p1_stats['+/-'],
        'PTS_2': p2_stats['PTS'],
        'FGM_2': p2_stats['FGM'],
        'FGA_2': p2_stats['FGA'],
        'FG%_2': p2_stats['FG%'],
        '3PM_2': p2_stats['3PM'],
        '3PA_2': p2_stats['3PA'],
        '3P%_2': p2_stats['3P%'],
        'FTM_2': p2_stats['FTM'],
        'FTA_2': p2_stats['FTA'],
        'FT%_2': p2_stats['FT%'],
        'OREB_2': p2_stats['OREB'],
        'DREB_2': p2_stats['DREB'],
        'REB_2': p2_stats['REB'],
        'AST_2': p2_stats['AST'],
        'STL_2': p2_stats['STL'],
        'BLK_2': p2_stats['BLK'],
        'TOV_2': p2_stats['TOV'],
        'PF_2': p2_stats['PF'],
        'differential_2': p2_stats['+/-']
    }

    df_new = pd.DataFrame([row])
    df_new['TEAM_1'] = label_encoder.transform(df_new['TEAM_1'])
    df_new['OPPONENT'] = label_encoder.transform(df_new['OPPONENT'])

    y_pred_new = classifier.predict(df_new)
    y_pred_new_decoded = ['W' if x == 1 else 'L' for x in y_pred_new]

    result.set(f"Predicted Outcome: {y_pred_new_decoded[0]}")

# GUI Layout
season_label = tk.Label(root, text="2023-24 SEASON", font=("Arial", 16))
season_label.grid(row=0, columnspan=2, pady=10)

team1_label = tk.Label(root, text="Home Team:")
team1_label.grid(row=1, column=0, padx=10, pady=5)
team1_entry = tk.Entry(root)
team1_entry.grid(row=1, column=1, padx=10, pady=5)

team2_label = tk.Label(root, text="Away Team:")
team2_label.grid(row=2, column=0, padx=10, pady=5)
team2_entry = tk.Entry(root)
team2_entry.grid(row=2, column=1, padx=10, pady=5)

analyze_button = tk.Button(root, text="Analyze", command=analyze_matchup)
analyze_button.grid(row=3, columnspan=2, pady=20)

result = tk.StringVar()
result_label = tk.Label(root, textvariable=result, font=("Arial", 14))
result_label.grid(row=4, columnspan=2, pady=10)

root.mainloop()
