#!/usr/bin/env python3
"""
Build JSON indexes of images by names using IMDb, writing output to:

    cache/multimedia/profiles.json
    cache/multimedia/posters.json
    cache/multimedia/backdrops.json

Requirements:
    pip install cinemagoer tqdm
"""

import os
import json
import time
from collections import defaultdict

from imdb import IMDb
from tqdm import tqdm  # progress bars


# ============================
# CONFIGURATION
# ============================

# Dataset root (read-only)
DATA_ROOT = "/space_mounts/atai-hs25/dataset"
IMAGES_JSON = os.path.join(DATA_ROOT, "additional", "images.json")

# Writable local cache folder
CACHE_ROOT = "./cache/multimedia"
os.makedirs(CACHE_ROOT, exist_ok=True)

OUTPUT_PROFILES = os.path.join(CACHE_ROOT, "profiles.json")
OUTPUT_POSTERS = os.path.join(CACHE_ROOT, "posters.json")
OUTPUT_BACKDROPS = os.path.join(CACHE_ROOT, "backdrops.json")

REQUEST_SLEEP = 0.2  # polite IMDb request delay


# ============================
# HELPERS
# ============================

def normalize_image_id(img: str) -> str:
    """Turn '0344/abc.jpg' → '0344/abc'."""
    return img[:-4] if img.lower().endswith(".jpg") else img


def collect_ids(images_json_path: str):
    """
    Scan images.json and group by IMDb IDs.

    Returns:
        profiles_by_pid
        posters_by_mid
        backdrops_by_mid
    """
    with open(images_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles_by_pid = defaultdict(list)
    posters_by_mid = defaultdict(list)
    backdrops_by_mid = defaultdict(list)

    print("[INFO] Scanning images.json ...")
    for item in tqdm(data, desc="Processing images.json", unit="img"):
        img = item.get("img")
        img_type = item.get("type")
        movies = item.get("movie", [])
        cast = item.get("cast", [])

        if not img or not img_type:
            continue

        image_id = normalize_image_id(img)

        if img_type == "profile":
            for pid in cast:
                profiles_by_pid[pid].append(image_id)

        elif img_type == "poster":
            for mid in movies:
                posters_by_mid[mid].append(image_id)

        elif img_type == "backdrop":
            for mid in movies:
                backdrops_by_mid[mid].append(image_id)

    return profiles_by_pid, posters_by_mid, backdrops_by_mid


def fetch_person_names(pids):
    """IMDb person lookup using tqdm."""
    ia = IMDb()
    pid_to_name = {}

    print("[INFO] Fetching person names from IMDb ...")

    for pid in tqdm(sorted(set(pids)), desc="Fetching persons", unit="person"):
        code = pid[2:] if pid.startswith("nm") else pid
        name = pid

        try:
            person = ia.get_person(code)
            if person:
                name = person.get("name", pid)
        except Exception as e:
            print(f"[WARN] Failed to fetch person {pid}: {e}")

        pid_to_name[pid] = name

        if REQUEST_SLEEP:
            time.sleep(REQUEST_SLEEP)

    print(f"[INFO] Done fetching {len(pid_to_name)} person names.")
    return pid_to_name


def fetch_movie_titles(mids):
    """IMDb movie lookup using tqdm."""
    ia = IMDb()
    mid_to_title = {}

    print("[INFO] Fetching movie titles from IMDb ...")

    for mid in tqdm(sorted(set(mids)), desc="Fetching movies", unit="movie"):
        code = mid[2:] if mid.startswith("tt") else mid
        title = mid

        try:
            movie = ia.get_movie(code)
            if movie:
                title = movie.get("title", mid)
        except Exception as e:
            print(f"[WARN] Failed to fetch movie {mid}: {e}")

        mid_to_title[mid] = title

        if REQUEST_SLEEP:
            time.sleep(REQUEST_SLEEP)

    print(f"[INFO] Done fetching {len(mid_to_title)} movie titles.")
    return mid_to_title


def build_named_mappings(profiles_by_pid, posters_by_mid, backdrops_by_mid):
    pids = list(profiles_by_pid.keys())
    mids = list(set(list(posters_by_mid.keys()) + list(backdrops_by_mid.keys())))

    print(f"[INFO] Unique person IDs: {len(pids)}")
    print(f"[INFO] Unique movie  IDs: {len(mids)}")

    pid_to_name = fetch_person_names(pids) if pids else {}
    mid_to_title = fetch_movie_titles(mids) if mids else {}

    profiles_named = defaultdict(list)
    for pid, imgs in profiles_by_pid.items():
        name = pid_to_name.get(pid, pid)
        profiles_named[name].extend(imgs)

    posters_named = defaultdict(list)
    for mid, imgs in posters_by_mid.items():
        title = mid_to_title.get(mid, mid)
        posters_named[title].extend(imgs)

    backdrops_named = defaultdict(list)
    for mid, imgs in backdrops_by_mid.items():
        title = mid_to_title.get(mid, mid)
        backdrops_named[title].extend(imgs)

    return (
        dict(profiles_named),
        dict(posters_named),
        dict(backdrops_named),
    )


# ============================
# MAIN
# ============================

def main():
    if not os.path.exists(IMAGES_JSON):
        raise FileNotFoundError(f"images.json not found at: {IMAGES_JSON}")

    print("[INFO] Collecting IMDb ids from images.json …")
    profiles_by_pid, posters_by_mid, backdrops_by_mid = collect_ids(IMAGES_JSON)

    print(f"[INFO] profiles_by_pid:  {len(profiles_by_pid)}")
    print(f"[INFO] posters_by_mid:   {len(posters_by_mid)}")
    print(f"[INFO] backdrops_by_mid: {len(backdrops_by_mid)}")

    profiles_named, posters_named, backdrops_named = build_named_mappings(
        profiles_by_pid, posters_by_mid, backdrops_by_mid
    )

    os.makedirs(CACHE_ROOT, exist_ok=True)

    with open(OUTPUT_PROFILES, "w", encoding="utf-8") as f:
        json.dump(profiles_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {OUTPUT_PROFILES}")

    with open(OUTPUT_POSTERS, "w", encoding="utf-8") as f:
        json.dump(posters_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {OUTPUT_POSTERS}")

    with open(OUTPUT_BACKDROPS, "w", encoding="utf-8") as f:
        json.dump(backdrops_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved → {OUTPUT_BACKDROPS}")


if __name__ == "__main__":
    main()