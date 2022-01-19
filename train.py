import pickle
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA

import xgboost as xgb

hero_names = pd.read_csv("dataset/hero_names.csv")
players = pd.read_csv("dataset/players.csv")
item_ids = pd.read_csv("dataset/item_ids.csv")
match = pd.read_csv("dataset/match.csv")


def find_item(_id):
    return item_lookup.get(_id, f"unk_{_id}")


# Data Cleaning
## Create a lookup table for heroes
print("Creating hero lookup table...")
hero_lookup = dict(zip(hero_names.hero_id, hero_names.localized_name))
hero_lookup[0] = "Unknown"
players["hero"] = players["hero_id"].map(hero_lookup)
players.head().iloc[:, -1]

## Create a lookup table for items
print("Creating item lookup table...")
item_lookup = dict(zip(item_ids.item_id, item_ids.item_name))
item_lookup[0] = "Unknown"

print("Mapping the item with the item_lookup table...")
players["item_0"] = players["item_0"].map(find_item)
players["item_1"] = players["item_1"].map(find_item)
players["item_2"] = players["item_2"].map(find_item)
players["item_3"] = players["item_3"].map(find_item)
players["item_4"] = players["item_4"].map(find_item)
players["item_5"] = players["item_5"].map(find_item)

# Binary encode the items
print("Binary encoding the items...")
item0 = pd.get_dummies(players["item_0"].fillna(0))
item1 = pd.get_dummies(players["item_1"].fillna(0))
item2 = pd.get_dummies(players["item_2"].fillna(0))
item3 = pd.get_dummies(players["item_3"].fillna(0))
item4 = pd.get_dummies(players["item_4"].fillna(0))
item5 = pd.get_dummies(players["item_5"].fillna(0))

# Since which slot doesn't matter, we can just sum up the columns
print("Summing up the item columns...")
player_items = item0.add(item1).add(item2).add(item3).add(item4).add(item5)

# Binary encoding the player's hero
print("Binary encoding the player's hero...")
player_heroes = pd.get_dummies(players["hero"])

print("Creating a different columns for each team...")
radiant_cols = list(map(lambda s: "radiant_" + s, player_heroes.columns.values))
dire_cols = list(map(lambda s: "dire_" + s, player_heroes.columns.values))
radiant_items_cols = list(
    map(lambda s: "radiant_" + str(s), player_items.columns.values)
)
dire_items_cols = list(map(lambda s: "dire_" + str(s), player_items.columns.values))

radiant_heroes = []
dire_heroes = []
radiant_items = []
dire_items = []

for _, _index in players.groupby("match_id").groups.items():
    radiant_heroes.append(player_heroes.iloc[_index][:5].sum().values)
    dire_heroes.append(player_heroes.iloc[_index][5:].sum().values)
    radiant_items.append(player_items.iloc[_index][:5].sum().values)
    dire_items.append(player_items.iloc[_index][5:].sum().values)

print("Converting the lists to dataframes...")
radiant_heroes = pd.DataFrame(radiant_heroes, columns=radiant_cols)
dire_heroes = pd.DataFrame(dire_heroes, columns=dire_cols)
radiant_items = pd.DataFrame(radiant_items, columns=radiant_items_cols)
dire_items = pd.DataFrame(dire_items, columns=dire_items_cols)


# Predictive Models
# Create a dataframe with all the features
print("Creating a dataframe with all the features...")
X = pd.concat([radiant_heroes, radiant_items, dire_heroes, dire_items], axis=1)
print(X.head())

y = match["radiant_win"].apply(lambda x: 1 if x else 0)
classes = ["radiant_win", "dire_win"]

# Using PCA to reduce the dimentionality of the data
# Reduce the features from 612 to 70 (heroes*team + heroes*team*items)
print("Using PCA to reduce the dimentionality of the data...")
pca_model = PCA(n_components=70)
X_pca = pca_model.fit_transform(X)

print("Using 42 as the seed...")
seed = 42
# Decisiion Tree Classifier
print("Using Decision Tree Classifier...")
dtc = DecisionTreeClassifier(random_state=seed, max_depth=10)
print(f"Cross-validation score: {cross_val_score(estimator=dtc, X=X_pca, y=y).mean()}")
dtc.fit(X_pca, y)

# Random Foreset Classifier
print("Using Random Forest Classifier...")
rfc = RandomForestClassifier(random_state=seed, n_estimators=100)
print(f"Cross-validation score: {cross_val_score(estimator=rfc, X=X_pca, y=y).mean()}")
rfc.fit(X_pca, y)

# XGBoost
print("Using XGBoost...")
xgbc = xgb.XGBClassifier(random_state=seed, use_label_encoder=False, verbosity=0)
print(f"Cross-validation score: {cross_val_score(estimator=xgbc, X=X_pca, y=y).mean()}")
xgbc.fit(X_pca, y)

# Saving the best model and pca
print("Saving the best model and pca...")
xgbc.save_model("models/xgbc.bin")
with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca_model, f)
