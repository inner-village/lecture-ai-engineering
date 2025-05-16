import os
import time
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#from great_expectations.data_context import EphemeralDataContext


class DataLoader:
    @staticmethod
    def load_titanic_data(path=None):
        if path:
            return pd.read_csv(path)
        else:
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)

    @staticmethod
    def preprocess_titanic_data(data):
        data = data.copy()
        drop_cols = [col for col in ["PassengerId", "Name", "Ticket", "Cabin"] if col in data.columns]
        if drop_cols:
            data.drop(drop_cols, axis=1, inplace=True)
        if "Survived" in data.columns:
            return data.drop("Survived", axis=1), data["Survived"]
        else:
            return data, None


class DataValidator:
    @staticmethod
    def validate_titanic_data(data):
        if not isinstance(data, pd.DataFrame):
            return False, [{"success": False, "error": "データはDataFrame形式である必要があります"}]

        required_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            return False, [{"success": False, "missing_columns": missing}]

        data["Age"] = data["Age"].fillna(data["Age"].median())
        data["Embarked"] = data["Embarked"].fillna("S")

        conditions = [
            data["Pclass"].isin([1, 2, 3]).all(),
            data["Sex"].isin(["male", "female"]).all(),
            data["Age"].between(0, 100).all(),
            data["Fare"].between(0, 600).all(),
            data["Embarked"].isin(["C", "Q", "S"]).all(),
        ]

        results = [{"success": cond, "result": None} for cond in conditions]
        return all(conditions), results

        success = all(conditions)

        results = [
            {"success": cond} for cond in conditions
        ]
        return success, results


class ModelTester:
    @staticmethod
    def create_preprocessing_pipeline():
        numeric = ["Age", "Fare", "SibSp", "Parch"]
        categorical = ["Pclass", "Sex", "Embarked"]

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ])

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        preprocessor = ModelTester.create_preprocessing_pipeline()
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_params)),
        ])
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        start = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def compare_with_baseline(metrics, baseline=0.75):
        return metrics["accuracy"] >= baseline


# テスト関数（pytestでの確認用）
def test_data_validation():
    data = DataLoader.load_titanic_data()
    X, _ = DataLoader.preprocess_titanic_data(data)
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗"

    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5
    success, _ = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データチェックに失敗"


def test_model_performance():
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    assert ModelTester.compare_with_baseline(metrics), f"精度が低い: {metrics['accuracy']}"
    assert metrics["inference_time"] < 1.0, f"推論時間が長い: {metrics['inference_time']}秒"


if __name__ == "__main__":
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")
    for r in results:
        print(f"成功: {r['success']}, 結果: {r.get('result') or r.get('error')}")

    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")
    ModelTester.save_model(model)
    print("ベースライン比較:", "合格" if ModelTester.compare_with_baseline(metrics) else "不合格")
