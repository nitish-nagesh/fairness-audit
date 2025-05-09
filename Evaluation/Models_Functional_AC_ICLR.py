import sys
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, DMatrix, train as xgb_train
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    categorical_columns = [
        "sex", "age_cat_25-45", "age_cat_Greaterthan45", "age_cat_Lessthan25",
        "race_African-American", "race_Caucasian", "c_charge_degree_F", "c_charge_degree_M"
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

def init_models() -> Dict[str, object]:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False)
    }

def train_with_checkpoints_exact(model, X_train, y_train, model_name, label, checkpoints=20):
    n_iterations = 1000
    check_iters = np.linspace(1, n_iterations, checkpoints, dtype=int)
    n_samples = X_train.shape[0]
    output_matrix = np.zeros((n_samples, checkpoints))

    if isinstance(model, LogisticRegression):
        for i, it in enumerate(check_iters):
            m = LogisticRegression(max_iter=it, solver='saga', warm_start=False)
            m.fit(X_train, y_train)
            output_matrix[:, i] = m.predict_proba(X_train)[:, 1]

    elif isinstance(model, XGBClassifier):
        dtrain = DMatrix(X_train, label=y_train)
        param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        for i, it in enumerate(check_iters):
            bst = xgb_train(param, dtrain, num_boost_round=it)
            output_matrix[:, i] = bst.predict(dtrain)

    else:
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_train)[:, 1]
        else:
            proba = model.decision_function(X_train)
        output_matrix = np.tile(proba.reshape(-1, 1), (1, checkpoints))

    filename = f"checkpoints_{model_name.replace(' ', '_')}_{label.replace(' ', '_')}.csv"
    np.savetxt(filename, output_matrix, delimiter=",", fmt="%.6f")

def evaluate_all(compas_arff_path, no_fairness_csv, with_fairness_csv, our_prompt_csv, ic_samples_csv) -> pd.DataFrame:
    datasets = {
        "TR-TR": compas_arff_path,
        "TR-TS No Fairness": no_fairness_csv,
        "TR-TS With Fairness": with_fairness_csv,
        "TR-TS Our Prompt": our_prompt_csv,
        "TR-TS IC Samples": ic_samples_csv,
        "TS-TS No Fairness": no_fairness_csv,
        "TS-TS With Fairness": with_fairness_csv,
        "TS-TS Our Prompt": our_prompt_csv,
        "TS-TS IC Samples": ic_samples_csv,
        "TS-TR No Fairness": compas_arff_path,
        "TS-TR With Fairness": compas_arff_path,
        "TS-TR Our Prompt": compas_arff_path,
        "TS-TR IC Samples": compas_arff_path,
    }

    all_results = {}

    for label, path in datasets.items():
        if path.endswith(".arff"):
            data, _ = arff.loadarff(path)
            df = preprocess(pd.DataFrame(data))
        else:
            df = preprocess(pd.read_csv(path))

        X = df.drop(columns=["two_year_recid"])
        y = df["two_year_recid"].astype(int)

        if label.startswith("TR-TR") or label.startswith("TS-TS"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        elif label.startswith("TR-TS"):
            train_data, _ = arff.loadarff(compas_arff_path)
            df_train = preprocess(pd.DataFrame(train_data))
            X_train = df_train.drop(columns=["two_year_recid"])
            y_train = df_train["two_year_recid"].astype(int)
            X_test = X
            y_test = y
        elif label.startswith("TS-TR"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            test_data, _ = arff.loadarff(compas_arff_path)
            df_test = preprocess(pd.DataFrame(test_data))
            X_test = df_test.drop(columns=["two_year_recid"])
            y_test = df_test["two_year_recid"].astype(int)

        models = init_models()
        for name, model in models.items():
            train_with_checkpoints_exact(model, X_train, y_train, name, label)

    return pd.DataFrame()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python model_dynamics.py <compas_arff_path> <no_fairness_csv> <with_fairness_csv> <our_prompt_csv> <ic_samples_csv>")
        sys.exit(1)

    compas_arff_path = sys.argv[1]
    no_fairness_csv = sys.argv[2]
    with_fairness_csv = sys.argv[3]
    our_prompt_csv = sys.argv[4]
    ic_samples_csv = sys.argv[5]

    evaluate_all(compas_arff_path, no_fairness_csv, with_fairness_csv, our_prompt_csv, ic_samples_csv)
    print("\n✅ Checkpoint-based training dynamics have been saved for all models and evaluation setups.")
