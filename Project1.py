import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Titanic-Dataset.csv'
df = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(df.head())

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

imputer = SimpleImputer(strategy='most_frequent')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()  # Reshape to 1D array

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 6))

# 1. Histogram of Age Distribution
plt.subplot(2, 2, 1)
plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 2. Count Plot of Survived
plt.subplot(2, 2, 2)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')

# 3. Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.subplot(2, 2, 3)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), feature_names[indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Ranking')
plt.gca().invert_yaxis()

# 4. Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.subplot(2, 2, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
