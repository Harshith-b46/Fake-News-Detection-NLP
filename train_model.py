import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


print("Loading dataset...")

data = pd.read_csv("WELFake_Dataset.csv")
data = data.dropna()

# shuffle dataset
data = shuffle(data, random_state=42)

# combine title + text
data["content"] = data["title"] + " " + data["text"]
data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Vectorizing...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=3,
    max_features=80000,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training models...")

# ---------------- Logistic Regression ----------------
lr = LogisticRegression(
    max_iter=3000,
    class_weight='balanced'
)

lr.fit(X_train_vec, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_vec))


# ---------------- Naive Bayes ----------------
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test_vec))


# ---------------- SVM ----------------
svm = LinearSVC(class_weight='balanced')

svm = CalibratedClassifierCV(svm)

svm.fit(X_train_vec, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_vec))


print("\nModel Accuracy:")
print("Logistic Regression:", lr_acc)
print("Naive Bayes:", nb_acc)
print("SVM:", svm_acc)


# ---------------- Pick Best ----------------
models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "SVM": svm
}

scores = {
    "Logistic Regression": lr_acc,
    "Naive Bayes": nb_acc,
    "SVM": svm_acc
}

best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# ---------------- Save ----------------
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Best model saved!")