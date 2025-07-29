import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    'email_text': [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Meeting reminder for tomorrow at 10 AM.",
        "Urgent: Your account has been compromised. Verify immediately.",
        "Hello, hope you are doing well.",
        "Get rich quick! Invest now and earn millions.",
        "Review of Q3 financial performance attached.",
        "Claim your lottery prize now!",
        "Regarding the project deadline.",
        "Viagra 100mg - special offer!",
        "Lunch plans for Friday?",
        "Your package is on its way. Track it here.",
        "Buy now and get 50% off!",
        "Newsletter update from our team.",
        "Free credit score check - sign up today!",
        "Please find the report attached.",
        "You are selected for a cash prize! Reply to claim.",
        "Limited time offer! Buy now and save.",
        "Project update for next week.",
        "Exclusive discount just for you!",
        "Can we schedule a call for next Monday?",
        "Click here to unsubscribe from spam.",
        "Your order has been shipped.",
        "Huge winnings awaiting you!",
        "Monthly report for sales performance.",
        "Don't miss out on this incredible deal!",
        "Friendly reminder: invoice due soon.",
        "Win cash prizes instantly!",
        "Important security alert regarding your login.",
        "Confirm your subscription here.",
        "Online pharmacy deals."
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'spam', 'ham', 'spam'
    ]
}

df = pd.DataFrame(data)

print("--- Dataset Information ---")
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nLabel Distribution:")
print(df['label'].value_counts())
print("-" * 30)

df['label_encoded'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

X = df['email_text']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\n--- Data Split Information ---")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("-" * 30)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_transformed = tfidf_vectorizer.fit_transform(X_train)

X_test_transformed = tfidf_vectorizer.transform(X_test)

print(f"\n--- Feature Engineering ---")
print(f"Shape of X_train_transformed: {X_train_transformed.shape}")
print(f"Shape of X_test_transformed: {X_test_transformed.shape}")
print("-" * 30)

model = MultinomialNB()
model.fit(X_train_transformed, y_train)

print("\n--- Model Training ---")
print("Multinomial Naive Bayes model training complete.")
print("-" * 30)

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham (Predicted)', 'Spam (Predicted)'],
            yticklabels=['Ham (Actual)', 'Spam (Actual)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Spam Detection')
plt.show()

print("\nConfusion Matrix Details:")
print("True Negatives (TN):", cm[0, 0])
print("False Positives (FP):", cm[0, 1])
print("False Negatives (FN):", cm[1, 0])
print("True Positives (TP):", cm[1, 1])
print("-" * 30)

print("\n--- Testing with New, Unseen Emails ---")

def predict_email_type(email_text):
    email_transformed = tfidf_vectorizer.transform([email_text])
    prediction = model.predict(email_transformed)
    return "SPAM" if prediction[0] == 1 else "HAM"

new_emails_to_test = [
    "Hello team, the project status meeting is at 2 PM today.",
    "URGENT: Your bank account is suspended. Click to reactivate immediately to avoid fees.",
    "You've won a brand new car! Claim your prize now by clicking this link!",
    "Just checking in, how are you doing with the report?",
    "Free Bitcoin giveaway! Limited time offer, act fast to get your share.",
    "Your package has been delayed. Please update your shipping information.",
    "Confirm your email address to continue using our services."
]

for i, email in enumerate(new_emails_to_test):
    result = predict_email_type(email)
    print(f"Email {i+1}: '{email}'")
    print(f"Prediction: {result}\n")

print("-" * 30)
print("\nScript execution complete.")
