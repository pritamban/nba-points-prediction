import pandas as pd


def add_rolling_features(df):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Rolling scoring features
    df["PTS_L1"] = df["PTS"].shift(1)
    df["PTS_L3"] = df["PTS"].shift(1).rolling(3).mean()
    df["PTS_L5"] = df["PTS"].shift(1).rolling(5).mean()

    # Rolling minutes & shot volume
    df["MIN_L3"] = df["MIN"].shift(1).rolling(3).mean()
    df["FGA_L3"] = df["FGA"].shift(1).rolling(3).mean()

    return df


def add_context_features(df):
    # Home / Away from MATCHUP column
    df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    # Rest days
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days

    return df


def add_target(df):
    # Predict points in the NEXT game
    df["PTS_next_game"] = df["PTS"].shift(-1)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/jaylen_brown_game_logs.csv")

    df = add_rolling_features(df)
    df = add_context_features(df)
    df = add_target(df)

    # Drop rows with missing values from rolling windows
    df_model = df.dropna().reset_index(drop=True)

    df_model.to_csv("data/processed/jaylen_brown_model_data.csv", index=False)

    print("Feature engineering complete. Saved modeling dataset.")