import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Suppress pandas error_bad_lines warning
warnings.filterwarnings("ignore")

# Load CSV, skipping bad lines (works with older pandas)
data = pd.read_csv("data.csv", error_bad_lines=False)

print("Sample data loaded:")
print(data.head())

# Drop missing values
data = data.dropna()

# Map numeric strength to labels, only if 'strength' is integer
if data["strength"].dtype != 'O':
    data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})

print("\nRandom sample from cleaned data:")
print(data.sample(5))

# Function to split passwords into characters (for TfidfVectorizer)
def word(password):
    return list(password)

# Prepare features and labels
x = np.array(data["password"])
y = np.array(data["strength"])

# Vectorize passwords
tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

# Split into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.05, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

print(f"\nModel accuracy on test set: {model.score(xtest, ytest):.2f}")

# Prompt user for a password to check its strength (input will be visible)
user = input("\nEnter a password to check its strength: ")
user_vector = tdif.transform([user])
output = model.predict(user_vector)
print(f"Predicted password strength: {output[0]}")
