import pandas as pd
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud

nltk.download('stopwords')

# ----------------------
# 1. Load Dataset
# ----------------------
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------
# 2. Text Cleaning
# ----------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['message'] = df['message'].apply(clean_text)

# ----------------------
# 3. Vectorization
# ----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# ----------------------
# 4. Train-Test Split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
# 5. Model Comparison
# ----------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier()
}

best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
print(classification_report(y_test, best_model.predict(X_test)))

# ----------------------
# 6. Hyperparameter Tuning (Naive Bayes)
# ----------------------
params = {'alpha': [0.1, 0.5, 1.0]}
grid = GridSearchCV(MultinomialNB(), params, cv=5)
grid.fit(X_train, y_train)

print("\nBest NB Parameters:", grid.best_params_)
print("Best NB CV Score:", grid.best_score_)

# ----------------------
# 7. Confusion Matrix
# ----------------------
cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ----------------------
# 8. WordCloud
# ----------------------
spam_words = " ".join(df[df['label']==1]['message'])
wordcloud = WordCloud(width=800, height=400).generate(spam_words)

plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Spam WordCloud")
plt.show()

# ----------------------
# 9. Save Best Model
# ----------------------
joblib.dump(best_model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("\nModel saved successfully!")
