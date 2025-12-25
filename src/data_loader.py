import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

def get_player_id(player_name):
    all_players = players.find_players_by_full_name(player_name)
    if len(all_players) == 0:
        raise ValueError("Player not found")
    return all_players[0]["id"]

def load_game_logs(player_name, seasons):
    player_id = get_player_id(player_name)
    all_games = []

    for season in seasons:
        print(f"Fetching {season} season...")
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        df = gamelog.get_data_frames()[0]
        df["SEASON"] = season
        all_games.append(df)

        time.sleep(1)  # avoid API rate limits

    return pd.concat(all_games, ignore_index=True)


if __name__ == "__main__":
    seasons = ["2022-23", "2023-24", "2024-25"]
    df = load_game_logs("Jaylen Brown", seasons)

    df = df.sort_values("GAME_DATE")
    df.to_csv("data/raw/jaylen_brown_game_logs.csv", index=False)

    print("Saved Jaylen Brown game logs.")

