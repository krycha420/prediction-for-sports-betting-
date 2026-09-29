import requests
import time
import json
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


load_dotenv()

BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.getenv("BALLDONTLIE_API_KEY")
CSV_PATH = "nba_games_2020_2023.csv"
SEASONS = [2020, 2021, 2022, 2023]



def get_games(season: int, cursor: int = None):
    params = {"seasons[]": season, "per_page": 100}
    if cursor is not None:
        params["cursor"] = cursor
    headers = {"Authorization": f"Bearer {API_KEY}"}
    while True:
        resp = requests.get(f"{BASE_URL}/games", params=params, headers=headers)
        if resp.status_code ==429:
            print("Rate limit, wait")
            time.sleep(10)
            continue
        resp.raise_for_status()
        return resp.json()


def fetch_all_games(season: int):
    all_games = []
    cursor = None
    while True:
        time.sleep(1)
        response = get_games(season, cursor)
        all_games.extend(response["data"])
        print(f"Cursor {cursor}: łącznie {len(all_games)} meczów")
        next_cursor = response["meta"].get("next_cursor")
        if next_cursor is not None:
            cursor = next_cursor
        else:
            break

    return all_games
def fetch_multiple_seasons(seasons):
    all_games = []
    for season in seasons:
        print(f"Pobieram sezon {season}...")
        all_games.extend(fetch_all_games(season))
    return all_games


def flatten_game(game):
    return {
        "id": game["id"],
        "date": game["date"],
        "season": game["season"],
        "postseason": game["postseason"],
        "home_team": game["home_team"]["full_name"],
        "visitor_team": game["visitor_team"]["full_name"],
        "home_score": game["home_team_score"],
        "visitor_score": game["visitor_team_score"],
    }

def games_to_dataframe(games):
    data = []
    for game in games:
        flattened_game = flatten_game(game)
        data.append(flattened_game)
    return pd.DataFrame(data)
def to_team_perspective(df):
    home_rows = df.copy()
    home_rows = home_rows.rename(columns = {
        "home_team": "team",
        "visitor_team": "opponent",
        "home_score": "points_scored",
        "visitor_score": "points_allowed",
    })
    home_rows["is_home"] = True
    home_rows["won"] = home_rows["points_scored"]>home_rows["points_allowed"]
    visitor_rows = df.copy()
    visitor_rows = visitor_rows.rename(columns={
        "home_team": "opponent",
        "visitor_team": "team",
        "home_score": "points_allowed",
        "visitor_score": "points_scored",
    })
    visitor_rows["is_home"] = False
    visitor_rows["won"] = visitor_rows["points_scored"] > visitor_rows["points_allowed"]
    combined = pd.concat([home_rows, visitor_rows], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined





def add_rolling_features(team_df, window = 10):
    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)
    grouped = team_df.groupby("team")
    team_df[f"avg_points_scored_last{window}"] = (
        grouped["points_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    team_df[f"avg_points_allowed_last{window}"] = (
        grouped["points_allowed"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    team_df[f"win_rate_last{window}"] = (
        grouped["won"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )
    return team_df

if os.path.exists(CSV_PATH):
    print("Loading data from local copy")
    df = pd.read_csv(CSV_PATH)
else:
    print("No local CSV, loading data from API...")
    games = fetch_multiple_seasons(SEASONS)
    df = games_to_dataframe(games)
    df.to_csv(CSV_PATH, index=False)
def build_training_table(team_df, window = 10):
    feature_cols = ["id","date", "team", "opponent",
                    "is_home", f"avg_points_scored_last{window}",
                    f"avg_points_allowed_last{window}",
                    f"win_rate_last{window}",
                    "won"
                    ]
    home = team_df[team_df["is_home"]==True][feature_cols]
    away = team_df[team_df["is_home"] == False][feature_cols]
    merged = home.merge(away, on= "id", suffixes= ("_home", "_away"))
    return merged

def finalize_training_table(training_df):
    training_df = training_df.copy()
    training_df["home_win"] = training_df["won_home"].astype(int)
    training_df = training_df.rename(columns={"date_home": "date"})

    training_df = training_df.drop(columns = [
        "is_home_home", "is_home_away", "won_home", "won_away", "date_away"
    ])
    training_df = training_df.dropna()
    training_df["point_diff_home"] = (
            training_df["avg_points_scored_last10_home"] - training_df["avg_points_allowed_last10_home"]
    )
    training_df["point_diff_away"] = (
            training_df["avg_points_scored_last10_away"] - training_df["avg_points_allowed_last10_away"]
    )
    training_df["strength_gap"] = training_df["point_diff_home"] - training_df["point_diff_away"]
    training_df["win_rate_gap"] = training_df["win_rate_last10_home"] - training_df["win_rate_last10_away"]


    return training_df

feature_columns = [
    "avg_points_scored_last10_home",
    "avg_points_allowed_last10_home",
    "win_rate_last10_home",
    "avg_points_scored_last10_away",
    "avg_points_allowed_last10_away",
    "win_rate_last10_away",
    "point_diff_home",
    "point_diff_away",
    "strength_gap",
    "win_rate_gap",
]

def train_test_split_by_date(final_df, feature_cols, test_size = 0.2):
    split_idx = int(len(final_df)* (1- test_size))
    train_df = final_df[:split_idx]
    test_df = final_df[split_idx:]
    x_train = train_df[feature_cols]
    y_train = train_df["home_win"]
    x_test = test_df[feature_cols]
    y_test = test_df["home_win"]

    return x_train, y_train, x_test, y_test


to_team = to_team_perspective(df)
to_team = add_rolling_features(to_team, window = 10)

training_df = build_training_table(to_team, window=10)

final_df = finalize_training_table(training_df)
final_df = final_df.sort_values("date").reset_index(drop=True)

X_train, y_train,X_test, y_test = train_test_split_by_date(final_df, feature_columns)

print(f"y_train mean: {y_train.mean():.3f}")
print(f"y_test mean: {y_test.mean():.3f}")

model = HistGradientBoostingClassifier(
    random_state=42,
    max_depth=3,
    max_iter=100,
    learning_rate=0.05,
    min_samples_leaf=20,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
y_train_pred = model.predict(X_train)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"Log loss: {log_loss(y_test, y_proba):.3f}")
print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.3f}")