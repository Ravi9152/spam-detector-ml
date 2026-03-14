import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import joblib

# Step 1: Load dataset
df = pd.read_csv(r"D:\spam.csv", encoding="latin1")

# Remove unnecessary columns
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

# Rename columns for clarity
df.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

# Step 2: Encode labels (ham=0, spam=1)
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Step 4: Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate
print("Accuracy:", model.score(X_test_vec, y_test))

# Step 7: Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Step 8: Predict new email/SMS
def predict_message(message):
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    message_vec = vectorizer.transform([message])
    result = model.predict(message_vec)
    return "Spam" if result[0] == 1 else "Not Spam"

# Example usage
# print(predict_message("Hey, are we meeting at 5 PM today?"))
# print(predict_message("you are winner! claim your prize now"))
 