import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
scaler = StandardScaler()


df = pd.read_csv("atp_matches_2023.csv")

df['p1_1st_srv_%'] = (df['p1_1stIn']/df['p1_svpt'])*100
df['p2_1st_srv_%'] = (df['p2_1stIn']/df['p2_svpt'])*100
df['p1_1st_srv_won_%'] = (df['p1_1stWon']/df['p1_1stIn'])*100
df['p2_1st_srv_won_%'] = (df['p2_1stWon']/df['p2_1stIn'])*100

df.drop(columns=[], inplace=True)

#print(df.head())
#print(df.columns)

#need to create more features from raw features to make proper features for model training

# Assume df is your original DataFrame
categorical_data = df[['tourney_name', 'surface']]
categorical_encoded = encoder.fit_transform(categorical_data)
categorical_encoded_df = pd.DataFrame(categorical_encoded.toarray(), columns=encoder.get_feature_names_out(['tourney_name', 'surface']))

df.reset_index(drop=True, inplace=True)
df_encoded = pd.concat([df.drop(['tourney_name', 'surface'], axis=1), categorical_encoded_df], axis=1)

# Ensure column is created
specific_player_id = 12345
df_encoded['is_specific_player_winner'] = (df_encoded['p1_id'] == specific_player_id).astype(int)

# Proceed if column exists
if 'is_specific_player_winner' in df_encoded.columns:
    X = df_encoded.drop(['p1_name', 'p2_name', 'is_specific_player_winner'], axis=1)
    y = df_encoded['is_specific_player_winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
else:
    print("Column 'is_specific_player_winner' does not exist.")