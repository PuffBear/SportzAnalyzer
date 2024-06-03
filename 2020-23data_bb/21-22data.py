import requests
import pandas as pd

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