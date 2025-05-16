import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def test_model_accuracy():
    model = joblib.load("models/titanic_model.pkl")

    # データ準備（main.py と完全一致）
    df = pd.read_csv("day5/test/data/titanic.csv")
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Pclass"] = df["Pclass"].astype(float)
    df["Sex"] = df["Sex"].astype(float)
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)
    df["Survived"] = df["Survived"].astype(float)

    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    # main.py で使われる test_size=0.18, random_state=70 に合わせる（main.pyログ参照）
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.18, random_state=70)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", acc)
    assert acc >= 0.8, f"Accuracy too low: {acc}"
