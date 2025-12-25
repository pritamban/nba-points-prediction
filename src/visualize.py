import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

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
    return df.iloc[:split_idx], df.iloc[split_idx:]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/jaylen_brown_model_data.csv")

    train_df, test_df = time_train_test_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ----- Plot: Actual vs Predicted -----
    plt.figure()
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title("Actual vs Predicted Points (Jaylen Brown)")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()])
    plt.show()

        # ----- Plot: Error Distribution -----
    errors = preds - y_test

    plt.figure()
    plt.hist(errors, bins=20)
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.show()

        # ----- Plot: Feature Importance -----
    importances = model.feature_importances_
    sorted_idx = importances.argsort()

    plt.figure()
    plt.barh(
        [FEATURES[i] for i in sorted_idx],
        importances[sorted_idx]
    )
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.show()