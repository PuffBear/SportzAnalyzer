'''
This is a model that takes into account only service data, not return points. the accuracy of this 
model is 52% which is not a lot.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("atp_matches_2023.csv")
df = df.drop(columns=['player1_ht', 'player1_ioc', 'player1_age', 'player2_ht', 'player2_ioc', 'player2_age', 'score', 'best_of', 'round', 'minutes', 'p1_ace', 'p1_df', 'p2_ace', 'p2_df'])
df = df.drop(columns=['surface', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p2_1stIn', 'p2_svpt', 'p2_1stWon', 'p2_2ndWon', 'p1_bpSaved', 'p1_bpFaced', 'p2_bpSaved', 'p2_bpFaced'])
df.replace('#DIV/0!', pd.NA, inplace=True)
df = df.dropna()

label_encoder = LabelEncoder()
df['player1_name'] = label_encoder.fit_transform(df['player1_name'])
df['player2_name'] = label_encoder.fit_transform(df['player2_name'])
df['winner_name'] = label_encoder.fit_transform(df['winner_name'])

# Define features and target
X = df.drop(columns=['winner_name'])
y = df['winner_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


unique_labels = list(set(y_test))
target_names = label_encoder.inverse_transform(unique_labels)
#print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))