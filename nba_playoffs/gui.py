import tkinter as tk
from tkinter import messagebox
import pandas as pd
from model1 import classifier
from model1 import label_encoder
from sklearn.preprocessing import StandardScaler
from model1 import X_train

player_df = pd.read_csv('nba_playoff_team_data.csv')

def add_team_entries():
    row = len(entries) + 1
    home_entry = tk.Entry(frame, width=20)
    away_entry = tk.Entry(frame, width=20)
    home_entry.grid(row=row, column=0, padx=5, pady=5)
    away_entry.grid(row=row, column=1, padx=5, pady=5)
    entries.append((home_entry, away_entry))

def analyze():
    matches = [(entry[0].get(), entry[1].get()) for entry in entries]

    data = []

    for match in matches:
        team1, team2 = match
        # Get team1 stats
        p1_stats = player_df[player_df['TEAM'] == team1]
        if p1_stats.empty:
            print(f"Team {team1} not found in the dataset.")
            continue  # Skip to the next match if team1 is not found

        # Get team2 stats
        p2_stats = player_df[player_df['TEAM'] == team2]
        if p2_stats.empty:
            print(f"Team {team2} not found in the dataset.")
            continue  # Skip to the next match if team2 is not found

        # Extract the first row of stats for each team
        p1_stats = p1_stats.iloc[0]
        p2_stats = p2_stats.iloc[0]

        print(f"Stats for {team1}:", p1_stats)
        print(f"Stats for {team2}:", p2_stats)
        
        # Create a dictionary for the new DataFrame row
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
        
        # Append the row to the data list
        data.append(row)
    
    # Create the new DataFrame with the specified columns
    df_new = pd.DataFrame(data, columns=[
        'TEAM_1', 'OPPONENT', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-',
        'PTS_2', 'FGM_2', 'FGA_2', 'FG%_2', '3PM_2', '3PA_2', '3P%_2', 'FTM_2', 'FTA_2', 'FT%_2', 'OREB_2', 'DREB_2', 'REB_2', 'AST_2', 'STL_2', 'BLK_2', 'TOV_2', 'PF_2', 'differential_2'
    ])

    
    df_new['TEAM_1'] = label_encoder.fit_transform(df_new['TEAM_1'])
    df_new['OPPONENT'] = label_encoder.fit_transform(df_new['OPPONENT'])

    # Ensure the new data has the same features as the training data
    X_new = df_new

    # Predict the outcomes
    y_pred_new = classifier.predict(X_new)
    y_prob_new = classifier.predict_proba(X_new)

    # Decode the predicted outcomes
    y_pred_new_decoded = ['W' if x == 1 else 'L' for x in y_pred_new]
    # Display the predictions
    result_str = "\n".join([f"{match[0]} vs {match[1]}: {pred} (Probability: {prob[1]:.2f})" for match, pred, prob in zip(matches, y_pred_new_decoded, y_prob_new)])
    messagebox.showinfo("Predictions", result_str)



# Create the main window
root = tk.Tk()
root.title("NBA Match Analyzer")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Add initial team entry boxes
entries = []
add_team_entries()

# Add buttons
add_button = tk.Button(root, text="+", command=add_team_entries)
add_button.pack(pady=5)

analyze_button = tk.Button(root, text="Analyze", command=analyze)
analyze_button.pack(pady=5)

# Run the main loop
root.mainloop()