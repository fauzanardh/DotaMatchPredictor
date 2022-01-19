import os
import pickle
import xgboost
import pandas as pd

from sklearn.decomposition import PCA

import requests

hero_names = pd.read_csv("dataset/hero_names.csv")
item_ids = pd.read_csv("dataset/item_ids.csv")
default_X = pd.read_csv("default_X.csv")
default_X.loc[:, :] = 0
default_y = pd.read_csv("default_y.csv")
default_y.loc[:, :] = 0

# Data Cleaning
## Create a lookup table for heroes
print("Creating hero lookup table...")
hero_lookup = dict(zip(hero_names.hero_id, hero_names.localized_name))
hero_lookup[0] = "Unknown"

## Create a lookup table for items
print("Creating item lookup table...")
item_lookup = dict(zip(item_ids.item_id, item_ids.item_name))
item_lookup[0] = "Unknown"


def find_item(_id):
    return item_lookup.get(_id, f"unk_{_id}")


def parse_opendota_matches(match_id: int) -> pd.DataFrame:
    """
    :param match_id:
    :return:
    """
    url = f"https://api.opendota.com/api/matches/{match_id}?api_key={os.environ['OPENDOTA_API_KEY']}"
    r = requests.get(url)
    if r.status_code == 200:
        # Copy the default X and y
        X = default_X.copy()
        y = default_y.copy()

        data = r.json()
        players = data["players"]
        for player in players:
            # Parsing the player's hero
            hero_name = hero_lookup.get(player["hero_id"], f"unk_{player['hero_id']}")
            hero_name = (
                f"radiant_{hero_name}" if player["isRadiant"] else f"dire_{hero_name}"
            )
            X.loc[0, hero_name] = 1

            # Parsing the player's items
            for i in range(5):
                item_name = find_item(player["item_" + str(i)])
                item_name = (
                    f"radiant_{item_name}"
                    if player["isRadiant"]
                    else f"dire_{item_name}"
                )
                X.loc[0, item_name] += 1
        y.loc[0, "radiant_win"] = data["radiant_win"]
        return X, y
    else:
        raise Exception(f"Failed to get data from {url}")


print("Loading the model...")
model = xgboost.XGBClassifier()
model.load_model("models/xgbc.bin")

print("Loading the PCA...")
pca = pickle.load(open("models/pca.pkl", "rb"))

print("Parsing the match...")
X, y = parse_opendota_matches(1574041530)
print("Transforming the features to the PCA space...")
X = pca.transform(X)
print("Predicting...")
pred = model.predict(X)
winner = "Radiant" if pred[0] == 1 else "Dire"
actual_winner = "Radiant" if y.loc[0, "radiant_win"] == 1 else "Dire"
print(f"Predicted winner: {winner} (actual: {actual_winner})")
