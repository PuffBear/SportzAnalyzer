'''
This is a model that takes into account both service data and return points, but only of one player. the accuracy of this 
model is 89% which is very good. But the issue here is how do I make a dataframe on which the model can predict?
'''
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CleanedTopMatchStats.csv")

# Assuming 'winner' is the target and other columns are features
X = df.drop('winner', axis=1)  # Drop other non-feature columns as necessary
y = df['winner']  # Target variable

X_encoded = pd.get_dummies(X)
feature_columns = X_encoded.columns 

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
prediction_prob = classifier.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model, scaler, and feature columns for later use
joblib.dump(classifier, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_encoded.columns, 'feature_columns.pkl')  # Save the feature columns