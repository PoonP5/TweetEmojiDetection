
import os
#0	❤	_red_heart_
path = os.path.join("labels", "us_mapping.txt")

id_to_emoji = {}
with open(path, "r") as f:
    for line in f:
        idx, emoji, name = line.strip().split("\t")
        #print(idx, emoji, name)
        id_to_emoji[int(idx)] = emoji