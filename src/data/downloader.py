import os
import sys
from yt_dlp import YoutubeDL

input_file = sys.argv[1]
output_folder = sys.argv[2]

with open(input_file, "r") as f:
    urls = f.readlines()

print(f"Found {len(urls)} URLs in {input_file}")

DATA_DIR = f'./{output_folder}'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 720p only, model will use 360p bilinear upscaling as input
params = {"outtmpl": f"{DATA_DIR}/%(id)s", 
          "cachedir": DATA_DIR,
          "format": "bestvideo[height=720]+bestaudio", 
          "ignoreerrors": True}

with YoutubeDL(params) as ydl:
    ydl.download(urls)