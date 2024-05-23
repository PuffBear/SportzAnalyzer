import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("nba_regular2023_24.csv")
df = df.drop(columns=['MIN'])
df = df.dropna()
# Identify and replace non-numeric values
df.replace('-', np.nan, inplace=True)
df.dropna(inplace=True)
unique_teams = df["TEAM_1"].unique()

#print(df.head())
#print(df.columns)

label_encoder = LabelEncoder()
df['TEAM_1'] = label_encoder.fit_transform(df['TEAM_1'])
df['OPPONENT'] = label_encoder.fit_transform(df['OPPONENT'])
df['OUTCOME'] = label_encoder.fit_transform(df['OUTCOME'])

'''
'TEAM_1', 'OPPONENT', 'OUTCOME', 'PTS', 'FGM', 'FGA', 'FG%', '3PM',
       '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TOV', 'PF', '+/-', 'PTS_2', 'FGM_2', 'FGA_2', 'FG%_2', '3PM_2',
       '3PA_2', '3P%_2', 'FTM_2', 'FTA_2', 'FT%_2', 'OREB_2', 'DREB_2',
       'REB_2', 'AST_2', 'STL_2', 'BLK_2', 'TOV_2', 'PF_2', 'differential_2'
'''

# Define features and target
X = df.drop(columns=['OUTCOME'])
y = df['OUTCOME']

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
