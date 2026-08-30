import requests
import time
import json

BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = "aefebbe9-c636-426f-9a3f-b0a16d71d38c"


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
games = fetch_all_games(2023)
print(f"\nŁĄCZNIE: {len(games)} meczów")
print(json.dumps(games[0], indent=2))

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

def game_to_dataframe(games):
    for game in games