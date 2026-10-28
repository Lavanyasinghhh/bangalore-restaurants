"""Can we predict whether a restaurant is rated 4.0 or higher from its basic
attributes (cost, area, type, cuisine, online ordering, table booking)?

Votes are deliberately NOT used as a feature: they're collected at the same
time as the rating, so a new restaurant wouldn't have them. Using them would
make the model look better than it really is (target leakage).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["cost_for_two", "n_cuisines"]
BOOLEAN = ["online_order", "book_table"]
CATEGORICAL = ["location", "restaurant_type", "primary_cuisine"]


def build_dataset(df: pd.DataFrame):
    d = df.dropna(subset=["rating"]).copy()
    X = d[NUMERIC + BOOLEAN + CATEGORICAL].copy()
    X[BOOLEAN] = X[BOOLEAN].astype(int)
    y = (d["rating"] >= 4.0).astype(int)
    return X, y


def train(df: pd.DataFrame, seed: int = 42) -> dict:
    X, y = build_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC + BOOLEAN),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    names = model.named_steps["pre"].get_feature_names_out()
    importance = (pd.Series(model.named_steps["rf"].feature_importances_, index=names)
                    .sort_values(ascending=False).head(15))
    importance.index = importance.index.str.replace(r"^(num|cat)__", "", regex=True)
    return {
        "model": model,
        "roc_auc": roc_auc_score(y_test, proba),
        "baseline_positive_rate": y_test.mean(),
        "report": classification_report(y_test, pred, target_names=["below 4.0", "4.0+"]),
        "importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
