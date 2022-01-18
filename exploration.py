import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


# Load dataset from 'dataset/' directory
ability_ids = pd.read_csv("dataset/ability_ids.csv")
ability_upgrades = pd.read_csv("dataset/ability_upgrades.csv")
hero_names = pd.read_csv("dataset/hero_names.csv")
item_ids = pd.read_csv("dataset/item_ids.csv")
match_outcomes = pd.read_csv("dataset/match_outcomes.csv")
match = pd.read_csv("dataset/match.csv")
objectives = pd.read_csv("dataset/objectives.csv")
patch_dates = pd.read_csv("dataset/patch_dates.csv")
player_ratings = pd.read_csv("dataset/player_ratings.csv")
player_time = pd.read_csv("dataset/player_time.csv")
players = pd.read_csv("dataset/players.csv")
purchase_log = pd.read_csv("dataset/purchase_log.csv")
test_player = pd.read_csv("dataset/test_player.csv")

"""
**match.csv**

`tower_status_*` and `barracks_status_*` are binary masks indicating whether 
the structures has been destroyed or not.

For more details, you can visit https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails
"""
print(match.head())

"""
**players.csv**

Contains stats about players performance in individual matches
"""
print(players.head())

"""
**player_time.csv**

Contains exp, gold, and last hits for each player on a one minute intervals
"""
print(player_time.head())
print("Player time for match id 2:")
print(player_time.query("match_id == 2").T)


"""
**player_ratings.csv**

Contains trueskill ratings for players in the match.

True Skill is a rating method somewhat like Matchmaking Rating (MMR)
"""
print(player_ratings.head())

"""
**ability_upgrades.csv and ability_names.csv**

`ability_upgrades.csv` contains the upgrade of each skill performed at each level for each players, 
while `ability_ids.csv` contains the ability ids and the english names of the abilities.
"""
print(ability_upgrades.head())
print(ability_ids.head())

"""
**purchase_log.csv and item_ids.csv**

`purchase_log.csv` contains the time for each item purchases, 
while `item_ids.csv` contains the item ids and the english names of the items.
"""
print(purchase_log.head())
print(item_ids.head())


def find_item(_id):
    return item_lookup.get(_id, f"unk_{_id}")


# Data Cleaning
## Create a lookup table for heroes
hero_lookup = dict(zip(hero_names.hero_id, hero_names.localized_name))
hero_lookup[0] = "Unknown"
players["hero"] = players["hero_id"].map(hero_lookup)
players.head().iloc[:, -1]

## Create a lookup table for items
item_lookup = dict(zip(item_ids.item_id, item_ids.item_name))
item_lookup[0] = "Unknown"

players["item_0"] = players["item_0"].map(find_item)
players["item_1"] = players["item_1"].map(find_item)
players["item_2"] = players["item_2"].map(find_item)
players["item_3"] = players["item_3"].map(find_item)
players["item_4"] = players["item_4"].map(find_item)
players["item_5"] = players["item_5"].map(find_item)

print(players["item_0"].head())

# Binary encode the items
item0 = pd.get_dummies(players["item_0"].fillna(0))
item1 = pd.get_dummies(players["item_1"].fillna(0))
item2 = pd.get_dummies(players["item_2"].fillna(0))
item3 = pd.get_dummies(players["item_3"].fillna(0))
item4 = pd.get_dummies(players["item_4"].fillna(0))
item5 = pd.get_dummies(players["item_5"].fillna(0))

# Since which slot doesn't matter, we can just sum up the columns
player_items = item0.add(item1).add(item2).add(item3).add(item4).add(item5)

# Binary encoding the player's hero
player_heroes = pd.get_dummies(players["hero"])

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

radiant_heroes = pd.DataFrame(radiant_heroes, columns=radiant_cols)
dire_heroes = pd.DataFrame(dire_heroes, columns=dire_cols)
radiant_items = pd.DataFrame(radiant_items, columns=radiant_items_cols)
dire_items = pd.DataFrame(dire_items, columns=dire_items_cols)

# Check whether the dataset is split evenly between radiant win and dire win
sns.countplot(y="radiant_win", data=match)

# The data shows that there are a lot of items with unknown names,
# this is likely due to the fact that there are no item in the slot (each player/hero has 6 slot for items).
player_items.sum().sort_values(ascending=False).head(25).plot.bar()

# What are the most picked heroes?
top_25_heroes = player_heroes.sum().sort_values(ascending=False).head(25)
top_25_heroes.plot.bar()

# What are the win rate for the top 25 heroes?
# Calculate the win rate of each hero
name, count = zip(*top_25_heroes.items())
win_rates = []
for i in range(len(name)):
    hero_match_ids = players[players["hero"] == name[i]]["match_id"].tolist()
    win_rate = match[match["match_id"].isin(hero_match_ids)][
        "radiant_win"
    ].value_counts(normalize=True)[True]
    win_rates.append((name[i], win_rate))
hero_win_rates = pd.DataFrame(win_rates, columns=["hero", "win_rate"])

plt.xticks(rotation=90)
plt.ylim(0.5, 0.53)
sns.barplot(data=hero_win_rates, x="hero", y="win_rate")
