import pandas as pd 
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

url = "https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&ISTRound=&LeagueID=00&PlayerOrTeam=T&Season=2020-21&SeasonType=Regular%20Season&Sorter=DATE"

# Set headers to avoid being blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*"
}

# Make a GET request to fetch the raw data
response = requests.get(url, headers=headers)
data = response.json()

# Extract the headers and row set from the response
headers = data['resultSets'][0]['headers']
rows = data['resultSets'][0]['rowSet']

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)
'''
['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
       'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'VIDEO_AVAILABLE']
'''

df = df.drop(columns=['SEASON_ID', 'TEAM_ID', 'VIDEO_AVAILABLE'])

# Define a dictionary for renaming columns
rename_dict = {
    'TEAM_ABBREVIATION': 'TEAM',
    'GAME_DATE': 'GAME DATE',
    'MATCHUP': 'OPPONENT',
    'WL': 'OUTCOME',
    'MIN': 'MIN',
    'PTS': 'PTS',
    'FGM': 'FGM',
    'FGA': 'FGA',
    'FG_PCT': 'FG%',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FTM': 'FTM',
    'FTA': 'FTA',
    'FT_PCT': 'FT%',
    'OREB': 'OREB',
    'DREB': 'DREB',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    'TOV': 'TOV',
    'PF': 'PF',
    'PLUS_MINUS': '+/-'
}

# Rename the columns
df.rename(columns=rename_dict, inplace=True)

# Function to transform the data
def transform_data(df):
    games = []
    grouped = df.groupby('GAME_ID')

    for game_id, group in grouped:
        if len(group) != 2:
            print(f"Unexpected number of rows for GAME_ID {game_id}")
            continue

        team_1 = group.iloc[0]
        team_2 = group.iloc[1]

        game = {
            'TEAM_1': team_1['TEAM'],
            'OPPONENT': team_2['TEAM'],
            'GAME DATE': team_1['GAME DATE'],
            'OUTCOME': team_1['OUTCOME'],
            'MIN': team_1['MIN'],
            'PTS': team_1['PTS'],
            'FGM': team_1['FGM'],
            'FGA': team_1['FGA'],
            'FG%': team_1['FG%'],
            '3PM': team_1['3PM'],
            '3PA': team_1['3PA'],
            '3P%': team_1['3P%'],
            'FTM': team_1['FTM'],
            'FTA': team_1['FTA'],
            'FT%': team_1['FT%'],
            'OREB': team_1['OREB'],
            'DREB': team_1['DREB'],
            'REB': team_1['REB'],
            'AST': team_1['AST'],
            'STL': team_1['STL'],
            'BLK': team_1['BLK'],
            'TOV': team_1['TOV'],
            'PF': team_1['PF'],
            '+/-': team_1['+/-'],
            'PTS_2': team_2['PTS'],
            'FGM_2': team_2['FGM'],
            'FGA_2': team_2['FGA'],
            'FG%_2': team_2['FG%'],
            '3PM_2': team_2['3PM'],
            '3PA_2': team_2['3PA'],
            '3P%_2': team_2['3P%'],
            'FTM_2': team_2['FTM'],
            'FTA_2': team_2['FTA'],
            'FT%_2': team_2['FT%'],
            'OREB_2': team_2['OREB'],
            'DREB_2': team_2['DREB'],
            'REB_2': team_2['REB'],
            'AST_2': team_2['AST'],
            'STL_2': team_2['STL'],
            'BLK_2': team_2['BLK'],
            'TOV_2': team_2['TOV'],
            'PF_2': team_2['PF'],
            'differential_2': team_2['+/-']
        }
        games.append(game)

    return pd.DataFrame(games)

# Transform the data
transformed_df = transform_data(df)
transformed_df = transformed_df.drop(columns=['GAME DATE', 'MIN'])
transformed_df = transformed_df.dropna()
twentytwenty_to_twentyone_data = transformed_df

url = "https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&ISTRound=&LeagueID=00&PlayerOrTeam=T&Season=2021-22&SeasonType=Regular%20Season&Sorter=DATE"

# Set headers to avoid being blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*"
}

# Make a GET request to fetch the raw data
response = requests.get(url, headers=headers)
data = response.json()

# Extract the headers and row set from the response
headers = data['resultSets'][0]['headers']
rows = data['resultSets'][0]['rowSet']

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)
'''
['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
       'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'VIDEO_AVAILABLE']
'''

df = df.drop(columns=['SEASON_ID', 'TEAM_ID', 'VIDEO_AVAILABLE'])

# Define a dictionary for renaming columns
rename_dict = {
    'TEAM_ABBREVIATION': 'TEAM',
    'GAME_DATE': 'GAME DATE',
    'MATCHUP': 'OPPONENT',
    'WL': 'OUTCOME',
    'MIN': 'MIN',
    'PTS': 'PTS',
    'FGM': 'FGM',
    'FGA': 'FGA',
    'FG_PCT': 'FG%',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FTM': 'FTM',
    'FTA': 'FTA',
    'FT_PCT': 'FT%',
    'OREB': 'OREB',
    'DREB': 'DREB',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    'TOV': 'TOV',
    'PF': 'PF',
    'PLUS_MINUS': '+/-'
}

# Rename the columns
df.rename(columns=rename_dict, inplace=True)

# Function to transform the data
def transform_data(df):
    games = []
    grouped = df.groupby('GAME_ID')

    for game_id, group in grouped:
        if len(group) != 2:
            print(f"Unexpected number of rows for GAME_ID {game_id}")
            continue

        team_1 = group.iloc[0]
        team_2 = group.iloc[1]

        game = {
            'TEAM_1': team_1['TEAM'],
            'OPPONENT': team_2['TEAM'],
            'GAME DATE': team_1['GAME DATE'],
            'OUTCOME': team_1['OUTCOME'],
            'MIN': team_1['MIN'],
            'PTS': team_1['PTS'],
            'FGM': team_1['FGM'],
            'FGA': team_1['FGA'],
            'FG%': team_1['FG%'],
            '3PM': team_1['3PM'],
            '3PA': team_1['3PA'],
            '3P%': team_1['3P%'],
            'FTM': team_1['FTM'],
            'FTA': team_1['FTA'],
            'FT%': team_1['FT%'],
            'OREB': team_1['OREB'],
            'DREB': team_1['DREB'],
            'REB': team_1['REB'],
            'AST': team_1['AST'],
            'STL': team_1['STL'],
            'BLK': team_1['BLK'],
            'TOV': team_1['TOV'],
            'PF': team_1['PF'],
            '+/-': team_1['+/-'],
            'PTS_2': team_2['PTS'],
            'FGM_2': team_2['FGM'],
            'FGA_2': team_2['FGA'],
            'FG%_2': team_2['FG%'],
            '3PM_2': team_2['3PM'],
            '3PA_2': team_2['3PA'],
            '3P%_2': team_2['3P%'],
            'FTM_2': team_2['FTM'],
            'FTA_2': team_2['FTA'],
            'FT%_2': team_2['FT%'],
            'OREB_2': team_2['OREB'],
            'DREB_2': team_2['DREB'],
            'REB_2': team_2['REB'],
            'AST_2': team_2['AST'],
            'STL_2': team_2['STL'],
            'BLK_2': team_2['BLK'],
            'TOV_2': team_2['TOV'],
            'PF_2': team_2['PF'],
            'differential_2': team_2['+/-']
        }
        games.append(game)

    return pd.DataFrame(games)

# Transform the data
transformed_df = transform_data(df)
transformed_df = transformed_df.drop(columns=['GAME DATE', 'MIN'])
transformed_df = transformed_df.dropna()
twentyone_to_twentytwo_data = transformed_df

url = "https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&ISTRound=&LeagueID=00&PlayerOrTeam=T&Season=2022-23&SeasonType=Regular%20Season&Sorter=DATE"

# Set headers to avoid being blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*"
}

# Make a GET request to fetch the raw data
response = requests.get(url, headers=headers)
data = response.json()

# Extract the headers and row set from the response
headers = data['resultSets'][0]['headers']
rows = data['resultSets'][0]['rowSet']

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)
'''
['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID',
       'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
       'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'VIDEO_AVAILABLE']
'''

df = df.drop(columns=['SEASON_ID', 'TEAM_ID', 'VIDEO_AVAILABLE'])

# Define a dictionary for renaming columns
rename_dict = {
    'TEAM_ABBREVIATION': 'TEAM',
    'GAME_DATE': 'GAME DATE',
    'MATCHUP': 'OPPONENT',
    'WL': 'OUTCOME',
    'MIN': 'MIN',
    'PTS': 'PTS',
    'FGM': 'FGM',
    'FGA': 'FGA',
    'FG_PCT': 'FG%',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FTM': 'FTM',
    'FTA': 'FTA',
    'FT_PCT': 'FT%',
    'OREB': 'OREB',
    'DREB': 'DREB',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    'TOV': 'TOV',
    'PF': 'PF',
    'PLUS_MINUS': '+/-'
}

# Rename the columns
df.rename(columns=rename_dict, inplace=True)

# Function to transform the data
def transform_data(df):
    games = []
    grouped = df.groupby('GAME_ID')

    for game_id, group in grouped:
        if len(group) != 2:
            print(f"Unexpected number of rows for GAME_ID {game_id}")
            continue

        team_1 = group.iloc[0]
        team_2 = group.iloc[1]

        game = {
            'TEAM_1': team_1['TEAM'],
            'OPPONENT': team_2['TEAM'],
            'GAME DATE': team_1['GAME DATE'],
            'OUTCOME': team_1['OUTCOME'],
            'MIN': team_1['MIN'],
            'PTS': team_1['PTS'],
            'FGM': team_1['FGM'],
            'FGA': team_1['FGA'],
            'FG%': team_1['FG%'],
            '3PM': team_1['3PM'],
            '3PA': team_1['3PA'],
            '3P%': team_1['3P%'],
            'FTM': team_1['FTM'],
            'FTA': team_1['FTA'],
            'FT%': team_1['FT%'],
            'OREB': team_1['OREB'],
            'DREB': team_1['DREB'],
            'REB': team_1['REB'],
            'AST': team_1['AST'],
            'STL': team_1['STL'],
            'BLK': team_1['BLK'],
            'TOV': team_1['TOV'],
            'PF': team_1['PF'],
            '+/-': team_1['+/-'],
            'PTS_2': team_2['PTS'],
            'FGM_2': team_2['FGM'],
            'FGA_2': team_2['FGA'],
            'FG%_2': team_2['FG%'],
            '3PM_2': team_2['3PM'],
            '3PA_2': team_2['3PA'],
            '3P%_2': team_2['3P%'],
            'FTM_2': team_2['FTM'],
            'FTA_2': team_2['FTA'],
            'FT%_2': team_2['FT%'],
            'OREB_2': team_2['OREB'],
            'DREB_2': team_2['DREB'],
            'REB_2': team_2['REB'],
            'AST_2': team_2['AST'],
            'STL_2': team_2['STL'],
            'BLK_2': team_2['BLK'],
            'TOV_2': team_2['TOV'],
            'PF_2': team_2['PF'],
            'differential_2': team_2['+/-']
        }
        games.append(game)

    return pd.DataFrame(games)

# Transform the data
transformed_df = transform_data(df)
transformed_df = transformed_df.drop(columns=['GAME DATE', 'MIN'])
transformed_df = transformed_df.dropna()
twentytwo_to_twentythree_data = transformed_df

overall_data = pd.concat([twentytwenty_to_twentyone_data, twentyone_to_twentytwo_data, twentytwo_to_twentythree_data], ignore_index=True)
print(overall_data)
overall_data = overall_data.dropna()

label_encoder = LabelEncoder()
overall_data['TEAM_1'] = label_encoder.fit_transform(overall_data['TEAM_1'])
overall_data['OPPONENT'] = label_encoder.fit_transform(overall_data['OPPONENT'])
overall_data['OUTCOME'] = label_encoder.fit_transform(overall_data['OUTCOME'])

'''
'TEAM_1', 'OPPONENT', 'OUTCOME', 'PTS', 'FGM', 'FGA', 'FG%', '3PM',
       '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', '+/-', 'PTS_2', 'FGM_2', 'FGA_2', 'FG%_2', '3PM_2',
       '3PA_2', '3P%_2', 'FTM_2', 'FTA_2', 'FT%_2', 'OREB_2', 'DREB_2',
       'REB_2', 'AST_2', 'STL_2', 'BLK_2', 'TOV_2', 'PF_2', 'differential_2'
'''

# Define features and target
X = overall_data.drop(columns=['OUTCOME'])
y = overall_data['OUTCOME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob_pred = classifier.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred_decoded = label_encoder.inverse_transform(y_pred)

#print(X_test)
print(y_prob_pred)
print(y_pred_decoded)

# Define the URL
url = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2022-23&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="

# Set headers to avoid being blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*"
}

# Make a GET request to fetch the raw HTML content
response = requests.get(url, headers=headers)
data = response.json()

# Extract the headers and row set from the response
headers = data['resultSets'][0]['headers']
rows = data['resultSets'][0]['rowSet']

df = pd.DataFrame(rows, columns=headers)

'''
['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA',
       'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB',
       'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS',
       'PLUS_MINUS', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
       'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
       'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK',
       'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK',
       'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK']
'''

df = df.drop(columns=['TEAM_ID', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
       'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
       'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK',
       'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK',
       'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'BLKA', 'PFD', ])

# Define a dictionary for renaming columns
rename_dict = {
    'TEAM_NAME': 'TEAM',
    'GAME_DATE': 'GAME DATE',
    'MATCHUP': 'OPPONENT',
    'WL': 'OUTCOME',
    'MIN': 'MIN',
    'PTS': 'PTS',
    'FGM': 'FGM',
    'FGA': 'FGA',
    'FG_PCT': 'FG%',
    'FG3M': '3PM',
    'FG3A': '3PA',
    'FG3_PCT': '3P%',
    'FTM': 'FTM',
    'FTA': 'FTA',
    'FT_PCT': 'FT%',
    'OREB': 'OREB',
    'DREB': 'DREB',
    'REB': 'REB',
    'AST': 'AST',
    'STL': 'STL',
    'BLK': 'BLK',
    'TOV': 'TOV',
    'PF': 'PF',
    'PLUS_MINUS': '+/-'
}

# Rename the columns
df.rename(columns=rename_dict, inplace=True)

player_df = df
print(player_df)

# Initialize the data for the new DataFrame
data = []

# Example matches
matches = [
    ('Houston Rockets', 'San Antonio Spurs'),
    ('Brooklyn Nets', 'Dallas Mavericks'),
    ('LA Clippers', 'Utah Jazz'),
    ('Orlando Magic', 'Portland Trail Blazers'),
    ('Golden State Warriors', 'Sacramento Kings')
]

# Ensure column names match exactly with those in 'player_df'
required_columns = [
    'TEAM', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'
]

# Check if all required columns are present in player_df
if not all(col in player_df.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in player_df.columns]
    raise ValueError(f"Missing columns in player_df: {missing_cols}")

for match in matches:
    team1 = match[0]
    team2 = match[1]
    
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

# Display the new DataFrame
print(df_new)

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
print("Predicted Outcomes:", y_pred_new_decoded)
print("Prediction Probabilities:", y_prob_new)