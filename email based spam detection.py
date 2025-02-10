import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Downloaded the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download
# Ensure the file name is 'spam.csv' and it is in the current directory.
data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['Label', 'EmailText']

# Convert labels to binary: 'spam' -> 1, 'ham' -> 0
data['Label'] = data['Label'].map({'spam': 1, 'ham': 0})

X = data['EmailText']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def predict_email_spam(email_content):
    email_vector = vectorizer.transform([email_content])
    prediction = model.predict(email_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test with a sample email
sample_email = "Congratulations! You've won a Rs.1,000 Reliance gift card. Click here to claim now."
print(f"\nCustom Email Prediction: {predict_email_spam(sample_email)}")
