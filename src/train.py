import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURES = [
    "PTS_L1",
    "PTS_L3",
    "PTS_L5",
    "MIN_L3",
    "FGA_L3",
    "REST_DAYS",
    "HOME"
]

TARGET = "PTS_next_game"


def time_train_test_split(df, train_frac=0.75):
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


if __name__ == "__main__":
    df = pd.read_csv("data/processed/jaylen_brown_model_data.csv")

    X = df[FEATURES]
    y = df[TARGET]

    train_df, test_df = time_train_test_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # ----- Baseline Model -----
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_preds = lin_reg.predict(X_test)

    lin_mae = mean_absolute_error(y_test, lin_preds)

    # ----- Random Forest Model -----
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_preds)

    print("Model Performance (MAE)")
    print("----------------------")
    print(f"Linear Regression MAE: {lin_mae:.2f}")
    print(f"Random Forest MAE:     {rf_mae:.2f}")
