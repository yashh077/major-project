import os
import json
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_suppo
rt, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import joblib

warnings.filterwarnings("ignore")


def load_dataset(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"Dataset not found at {csv_path}")
	return pd.read_csv(csv_path)


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
	])

	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		]
	)

	return preprocessor


def train_salary_regression(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
	# Target: Salary_PPP_Adjusted to reduce geo bias
	target_col = "Salary_PPP_Adjusted"
	feature_cols_num = [
		"Age",
		"Education_Level_Code",
		"Years of Experience",
		"Senior",
		"PPP_Index",
	]
	feature_cols_cat = [
		"Gender",
		"Education_Level",
		"Job Title",
		"Country",
		"Race",
	]

	data = df.copy()
	data = data.replace({np.inf: np.nan, -np.inf: np.nan})
	data = data.dropna(subset=[target_col])

	X = data[feature_cols_num + feature_cols_cat]
	y = data[target_col].astype(float)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	preprocessor = build_preprocessor(feature_cols_num, feature_cols_cat)
	regressor = GradientBoostingRegressor(
		n_estimators=600,
		learning_rate=0.05,
		max_depth=3,
		random_state=42,
	)

	model = Pipeline(steps=[("preprocess", preprocessor), ("regressor", regressor)])
	model.fit(X_train, y_train)

	# Evaluate
	y_pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, y_pred)

	metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "n_train": len(X_train), "n_test": len(X_test)}
	return model, metrics


def train_fake_classification(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
	target_col = "Fake_Job_Risk"
	feature_cols_num = [
		"Age",
		"Education_Level_Code",
		"Years of Experience",
		"Senior",
		"Salary_PPP_Adjusted",
		"PPP_Index",
	]
	feature_cols_cat = [
		"Gender",
		"Education_Level",
		"Job Title",
		"Country",
		"Race",
	]

	data = df.copy()
	data = data.replace({np.inf: np.nan, -np.inf: np.nan})
	data = data.dropna(subset=[target_col])

	X = data[feature_cols_num + feature_cols_cat]
	y = data[target_col].astype(int)

	# Stratify if possible
	stratify = y if y.nunique() == 2 else None
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=stratify
	)

	preprocessor = build_preprocessor(feature_cols_num, feature_cols_cat)
	classifier = GradientBoostingClassifier(
		n_estimators=600,
		learning_rate=0.05,
		max_depth=3,
		random_state=42,
	)

	model = Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
	model.fit(X_train, y_train)

	# Evaluate
	y_prob = model.predict_proba(X_test)[:, 1]
	y_pred = (y_prob >= 0.5).astype(int)
	roc = roc_auc_score(y_test, y_prob)
	ap = average_precision_score(y_test, y_prob)
	precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
	acc = accuracy_score(y_test, y_pred)

	metrics = {
		"ROC_AUC": roc,
		"PR_AUC": ap,
		"Precision": precision,
		"Recall": recall,
		"F1": f1,
		"Accuracy": acc,
		"n_train": len(X_train),
		"n_test": len(X_test),
	}
	return model, metrics


def main():
	csv_path = "finalDataset.csv"
	print(f"Loading dataset from {csv_path} ...")
	df = load_dataset(csv_path)
	print(f"Loaded shape: {df.shape}")

	# Train salary regressor
	print("Training salary regression (target=Salary_PPP_Adjusted)...")
	salary_model, reg_metrics = train_salary_regression(df)
	print("Regression metrics:", json.dumps(reg_metrics, indent=2))

	# Train fake job classifier
	print("Training fake post classifier (target=Fake_Job_Risk)...")
	fake_model, clf_metrics = train_fake_classification(df)
	print("Classification metrics:", json.dumps(clf_metrics, indent=2))

	# Save models
	joblib.dump(salary_model, "salary_regressor.pkl")
	joblib.dump(fake_model, "fake_classifier.pkl")
	print("Saved models: salary_regressor.pkl, fake_classifier.pkl")


if __name__ == "__main__":
	main()


