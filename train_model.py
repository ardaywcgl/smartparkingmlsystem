import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

def train():
    """
    Executes the model training process: 
    Data loading, cleaning, feature engineering, and model saving.
    """
    dataset_path = "dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(dataset_path)
    
    # Specific parking lot selection (Example: BHMBCCMKT01)
    park_id = "BHMBCCMKT01"
    df = df[df["SystemCodeNumber"] == park_id].copy()
    
    df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
    df = df.sort_values("LastUpdated")

    # Target Variable: Occupancy Rate (Ratio)
    df["occupancy_rate"] = df["Occupancy"] / df["Capacity"]

    # Feature Engineering (Data Leakage Fix - Shift 1)
    df["rolling_3h"] = df["occupancy_rate"].shift(1).rolling(window=3, min_periods=1).mean()
    df["rolling_24h"] = df["occupancy_rate"].shift(1).rolling(window=24, min_periods=1).mean()
    
    # Baseline: Occupancy at the same hour yesterday (For comparison)
    df["baseline_yesterday"] = df["occupancy_rate"].shift(24) 

    # Clean missing data
    df = df.dropna()

    # Time-based features
    df["hour"] = df["LastUpdated"].dt.hour
    df["day_of_week"] = df["LastUpdated"].dt.dayofweek
    df["month"] = df["LastUpdated"].dt.month
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["week_of_year"] = df["LastUpdated"].dt.isocalendar().week.astype(int)

    features = ["hour", "day_of_week", "month", "is_weekend", "week_of_year", "rolling_3h", "rolling_24h"]
    X = df[features]
    y = df["occupancy_rate"]

    # Chronologically split data (Train: 80%, Test: 20%)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training model ({len(X_train)} samples)...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    
    # Baseline MAE
    baseline_mae = mean_absolute_error(y_test, df["baseline_yesterday"].iloc[split_idx:])
    
    print("-" * 30)
    print(f"Model MAE: {model_mae:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    print(f"Improvement Rate: {improvement:.2f}%")
    print("-" * 30)
    
    # Saving
    joblib.dump(model, "parking_model.joblib")
    metadata = {
        "mae": model_mae,
        "baseline_mae": baseline_mae,
        "last_rolling_3h": df["rolling_3h"].iloc[-1],
        "last_rolling_24h": df["rolling_24h"].iloc[-1],
        "features": features
    }
    joblib.dump(metadata, "model_metadata.joblib")
    print("Model and metadata saved successfully.")

if __name__ == "__main__":
    train()