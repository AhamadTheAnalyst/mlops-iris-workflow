# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

RANDOM_STATE = 42
N_ESTIMATORS = 200  # change from 100

def main():
    # 1) Data
    X, y = load_iris(return_X_y=True)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3) Model
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 4) Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Saved with n_estimators={N_ESTIMATORS}")

    # 5) Save model (ignored by git via .gitignore)
    joblib.dump(clf, "model.pkl")

if __name__ == "__main__":
    main()
