import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


class ForecastEngine:
    def __init__(self, model, target_col, feature_cols, metrics):
        self.model = model
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.metrics = metrics
        self.last_X = None  # store last input for estimation

    # ===============================
    # TRAIN MODEL
    # ===============================
    @classmethod
    def auto_train(cls, df: pd.DataFrame, target_col: str):
        df = df.copy()

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        num_features = X.select_dtypes(include="number").columns.tolist()
        # 🚫 NEVER USE DATE AS FEATURE
        X = X.loc[:, ~X.columns.str.contains("date", case=False)]
        num_features = X.select_dtypes(include="number").columns.tolist()
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )

        model = Pipeline(steps=[
            ("prep", preprocessor),
            ("reg", RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "r2": round(r2_score(y_test, preds), 3),
            "mae": round(mean_absolute_error(y_test, preds), 2)
        }

        engine = cls(
            model=model,
            target_col=target_col,
            feature_cols=list(X.columns),
            metrics=metrics
        )

        # store last known data for future estimation
        engine.last_X = X.tail(1)

        return engine

    # ===============================
    # RUN ESTIMATION
    # ===============================
    def predict(self, horizon: int = 1):
        if horizon < 1:
            raise ValueError("Horizon must be >= 1")

        if self.last_X is None:
            raise ValueError("Model has no reference data")

        predictions = []

        # reuse last known row as proxy for future
        for _ in range(horizon):   # ✅ horizon is an INTEGER
            pred = self.model.predict(self.last_X)[0]
            predictions.append(pred)

        return {
            "target": self.target_col,
            "horizon": horizon,
            "mean_prediction": round(float(np.mean(predictions)), 2),
            "min_prediction": round(float(np.min(predictions)), 2),
            "max_prediction": round(float(np.max(predictions)), 2),
            "model_metrics": self.metrics,
            "trend": "stable"  # simple placeholder
        }
