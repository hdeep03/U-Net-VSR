import random
import textwrap
from yt_dlp import YoutubeDL

train_metadata_path = "./data/metadata/train.in"
val_metadata_path = "./data/metadata/val.in"
test_metadata_path = "./data/metadata/test.in"

# playlists = ["https://www.youtube.com/watch?v=dyywO6ORkiU&list=PLeCPQtbuJaPtLugP7UijMU-ozdNmkdNFO"]
playlists = ["https://www.youtube.com/watch?v=rZGKG5Owtbw&list=PLeCPQtbuJaPsD-TBP4QqBAHCJttYd4ABw"]

class loggerOutputs:
    def error(msg):
        print("Captured Error: "+msg)
    def warning(msg):
        pass
    def debug(msg):
        pass

ids = []

for playlist in playlists:

    with YoutubeDL({"ignoreerrors": True}) as ydl:
        playlist_dict = ydl.extract_info(playlist, download=False)

    # Pretty-printing the video information (optional)
    for video in playlist_dict["entries"]:
        if not video:
            print("ERROR: Unable to get info. Continuing...")
            continue
        if "id" in video:
            ids.append(video["id"])

print(ids)
random.shuffle(ids)
ids = ids[:100] # use first 100 videos for now
train_ids = ids[:int(0.8 * len(ids))]
val_ids = ids[int(0.8 * len(ids)):int(0.9*len(ids))]
test_ids = ids[int(0.9 * len(ids)):]
print(len(train_ids))
print(len(val_ids))
print(len(test_ids))

with open(train_metadata_path, "w") as f:
    f.write("\n".join(train_ids))

with open(val_metadata_path, "w") as f:
    f.write("\n".join(val_ids))

with open(test_metadata_path, "w") as f:
    f.write("\n".join(test_ids))
