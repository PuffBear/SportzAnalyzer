import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('nba_playoff_matchups.csv')

# Drop the 'GAME DATE' column and rows with missing values
df = df.drop(columns='GAME DATE')
df = df.dropna()

# Define features and target without encoding
X = df.drop(columns=['OUTCOME'])
y = df['OUTCOME']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of X_train, X_test, y_train, and y_test
print("X_train:")
print(X_train.head())

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train.head())

print("\ny_test:")
print(y_test)
