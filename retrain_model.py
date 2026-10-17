"""
retrain_model.py
----------------
Retrains the sentiment analysis model and vectorizer from the reviews dataset
and saves them as nlp_model.pkl and tranform.pkl.

Run this script whenever your scikit-learn version differs from the one used
to generate the existing .pkl files (e.g. after upgrading scikit-learn).

Usage:
    python retrain_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)

DATASET_PATH = os.path.join("datasets", "reviews.txt")

print("Loading dataset...")
dataset = pd.read_csv(DATASET_PATH, sep="\t", names=["Reviews", "Comments"])
print(f"  {len(dataset)} rows loaded.")

# sklearn 1.x requires stop_words to be a list, not a set
stop_words = list(stopwords.words("english"))

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    use_idf=True,
    lowercase=True,
    strip_accents="ascii",
    stop_words=stop_words,
)
X = vectorizer.fit_transform(dataset.Comments)
y = dataset.Reviews

# Quick accuracy check on a held-out split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
clf_eval = MultinomialNB()
clf_eval.fit(X_train, y_train)
acc = accuracy_score(y_test, clf_eval.predict(X_test)) * 100
print(f"  Test accuracy: {acc:.2f}%")

# Train final model on the full dataset
print("Training final model on full dataset...")
clf = MultinomialNB()
clf.fit(X, y)

# Save
pickle.dump(vectorizer, open("tranform.pkl", "wb"))
pickle.dump(clf, open("nlp_model.pkl", "wb"))
print("Saved: tranform.pkl, nlp_model.pkl")

# Sanity checks
pos = vectorizer.transform(np.array(["this movie was absolutely amazing"]))
neg = vectorizer.transform(np.array(["terrible boring awful waste of time"]))
print(f"  Sanity check (positive): {'Positive' if clf.predict(pos)[0] else 'Negative'}")
print(f"  Sanity check (negative): {'Positive' if clf.predict(neg)[0] else 'Negative'}")
print("Done.")
