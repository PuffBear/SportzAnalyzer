import streamlit as st
import plotly.express as px
import pandas as pd
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("nba_playoff_matchups.csv")
df = df.drop(columns=['GAME DATE','MIN'])
df = df.dropna()
unique_teams = df["TEAM_1"].unique()
label_encoder = LabelEncoder()
df['TEAM_1'] = label_encoder.fit_transform(df['TEAM_1'])
df['OPPONENT'] = label_encoder.fit_transform(df['OPPONENT'])
df['OUTCOME'] = label_encoder.fit_transform(df['OUTCOME'])
X = df.drop(columns=['OUTCOME'])
y = df['OUTCOME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_prob_pred = classifier.predict_proba(X_test)


player_df = pd.read_csv('nba_playoff_team_data.csv')
data = []
required_columns = [
    'TEAM', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'
]



teams = ["Select Team", "Boston Celtics" ,"Minnesota Timberwolves", "Dallas Mavericks", "Indiana Pacers", "Oklahoma City Thunder", "Denver Nuggets" ,"New York Knicks", "Orlando Magic", "Cleveland Cavaliers", "LA Clippers", "Milwaukee Bucks", "Philadelphia 76ers", "Los Angeles Lakers", "Miami Heat", "New Orleans Pelicans", "Phoenix Suns"]

st.set_page_config(layout="centered",page_title="Sports Analyzer", page_icon=":tennis:")
st.title("Common Oaks' Sports Analyzer")
st.header("NBA Games")

st.subheader("Choose the Teams that are playing: ")
# Team selection dropdowns
team1 = st.selectbox('Select Team 1:', teams)
team2 = st.selectbox('Select Team 2:', teams)

# Button to save the match
if st.button('Save Match'):
    match = [(team1, team2)]
    if not all(col in player_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in player_df.columns]
        raise ValueError(f"Missing columns in player_df: {missing_cols}")
    
    p1_stats = player_df[player_df['TEAM'] == team1]
    if p1_stats.empty:
        print(f"Team {team1} not found in the dataset.")

    # Get team2 stats
    p2_stats = player_df[player_df['TEAM'] == team2]
    if p2_stats.empty:
        print(f"Team {team2} not found in the dataset.")

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

    df_new = pd.DataFrame(data, columns=[
        'TEAM_1', 'OPPONENT', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-',
        'PTS_2', 'FGM_2', 'FGA_2', 'FG%_2', '3PM_2', '3PA_2', '3P%_2', 'FTM_2', 'FTA_2', 'FT%_2', 'OREB_2', 'DREB_2', 'REB_2', 'AST_2', 'STL_2', 'BLK_2', 'TOV_2', 'PF_2', 'differential_2'
    ])

    df_new['TEAM_1'] = label_encoder.fit_transform(df_new['TEAM_1'])
    df_new['OPPONENT'] = label_encoder.fit_transform(df_new['OPPONENT'])
    X_new = df_new
    y_pred_new = classifier.predict(X_new)
    y_prob_new = classifier.predict_proba(X_new)
    y_pred_new_decoded = ['W' if x == 1 else 'L' for x in y_pred_new]
    st.write("Predicted Outcomes:", y_pred_new_decoded)
    st.write("Prediction Probabilities:", y_prob_new)
