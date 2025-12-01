#!/usr/bin/env python3
"""
Build JSON indexes of images by names using IMDb, writing output
to a local writable folder: ./cache/

Output:
    cache/profiles.json
    cache/posters.json
    cache/backdrops.json

Requirements:
    pip install cinemagoer
"""

import os
import json
import time
from collections import defaultdict

from imdb import IMDb  # provided by cinemagoer / imdbpy


# ============================
# CONFIGURATION
# ============================

# Dataset location (read-only)
DATA_ROOT = "/space_mounts/atai-hs25/dataset"
IMAGES_JSON = os.path.join(DATA_ROOT, "additional", "images.json")

# Local writable cache directory
CACHE_ROOT = "./cache"
os.makedirs(CACHE_ROOT, exist_ok=True)

OUTPUT_PROFILES = os.path.join(CACHE_ROOT, "profiles.json")
OUTPUT_POSTERS = os.path.join(CACHE_ROOT, "posters.json")
OUTPUT_BACKDROPS = os.path.join(CACHE_ROOT, "backdrops.json")

REQUEST_SLEEP = 0.2  # polite delay between IMDb requests


# ============================
# HELPERS
# ============================

def normalize_image_id(img: str) -> str:
    """Turn '0344/abc.jpg' into '0344/abc'."""
    return img[:-4] if img.lower().endswith(".jpg") else img


def collect_ids(images_json_path: str):
    """
    First pass: read images.json and group IMDb IDs -> image_ids.

    Returns:
        profiles_by_pid:  dict[nm_id -> list[image_ids]]
        posters_by_mid:   dict[tt_id -> list[image_ids]]
        backdrops_by_mid: dict[tt_id -> list[image_ids]]
    """
    with open(images_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles_by_pid = defaultdict(list)
    posters_by_mid = defaultdict(list)
    backdrops_by_mid = defaultdict(list)

    for item in data:
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
    """IMDb person lookup: pid ('nm0000138') → name ('Leonardo DiCaprio')."""
    ia = IMDb()
    pid_to_name = {}

    for i, pid in enumerate(sorted(set(pids))):
        code = pid[2:] if pid.startswith("nm") else pid
        name = pid  # fallback

        try:
            person = ia.get_person(code)
            if person:
                name = person.get("name", pid)
        except Exception as e:
            print(f"[WARN] Failed to fetch person {pid}: {e}")

        pid_to_name[pid] = name

        if REQUEST_SLEEP:
            time.sleep(REQUEST_SLEEP)

        if (i + 1) % 100 == 0:
            print(f"[INFO] fetched {i+1} / {len(pids)} persons ...")

    print(f"[INFO] Person-name lookup finished ({len(pid_to_name)} entries).")
    return pid_to_name


def fetch_movie_titles(mids):
    """IMDb movie lookup: mid ('tt0109830') → movie title ('Forrest Gump')."""
    ia = IMDb()
    mid_to_title = {}

    for i, mid in enumerate(sorted(set(mids))):
        code = mid[2:] if mid.startswith("tt") else mid
        title = mid  # fallback

        try:
            movie = ia.get_movie(code)
            if movie:
                title = movie.get("title", mid)
        except Exception as e:
            print(f"[WARN] Failed to fetch movie {mid}: {e}")

        mid_to_title[mid] = title

        if REQUEST_SLEEP:
            time.sleep(REQUEST_SLEEP)

        if (i + 1) % 100 == 0:
            print(f"[INFO] fetched {i+1} / {len(mids)} movies ...")

    print(f"[INFO] Movie-title lookup finished ({len(mid_to_title)} entries).")
    return mid_to_title


def build_named_mappings(profiles_by_pid, posters_by_mid, backdrops_by_mid):
    # Unique IMDb IDs
    pids = list(profiles_by_pid.keys())
    mids = list(set(list(posters_by_mid.keys()) + list(backdrops_by_mid.keys())))

    print(f"[INFO] Unique person IDs: {len(pids)}")
    print(f"[INFO] Unique movie  IDs: {len(mids)}")

    # Fetch from IMDb
    pid_to_name = fetch_person_names(pids) if pids else {}
    mid_to_title = fetch_movie_titles(mids) if mids else {}

    # Convert ID groups → name groups
    profiles_named = defaultdict(list)
    for pid, images in profiles_by_pid.items():
        name = pid_to_name.get(pid, pid)
        profiles_named[name].extend(images)

    posters_named = defaultdict(list)
    for mid, images in posters_by_mid.items():
        title = mid_to_title.get(mid, mid)
        posters_named[title].extend(images)

    backdrops_named = defaultdict(list)
    for mid, images in backdrops_by_mid.items():
        title = mid_to_title.get(mid, mid)
        backdrops_named[title].extend(images)

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
        raise FileNotFoundError(f"images.json missing: {IMAGES_JSON}")

    print("[INFO] Collecting IDs from images.json …")
    profiles_by_pid, posters_by_mid, backdrops_by_mid = collect_ids(IMAGES_JSON)

    print(f"[INFO] profiles_by_pid:  {len(profiles_by_pid)}")
    print(f"[INFO] posters_by_mid:   {len(posters_by_mid)}")
    print(f"[INFO] backdrops_by_mid: {len(backdrops_by_mid)}")

    print("[INFO] Fetching names/titles from IMDb …")
    profiles_named, posters_named, backdrops_named = build_named_mappings(
        profiles_by_pid, posters_by_mid, backdrops_by_mid
    )

    # Ensure cache directory exists
    os.makedirs(CACHE_ROOT, exist_ok=True)

    # Write outputs
    with open(OUTPUT_PROFILES, "w", encoding="utf-8") as f:
        json.dump(profiles_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved profiles.json → {OUTPUT_PROFILES}")

    with open(OUTPUT_POSTERS, "w", encoding="utf-8") as f:
        json.dump(posters_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved posters.json → {OUTPUT_POSTERS}")

    with open(OUTPUT_BACKDROPS, "w", encoding="utf-8") as f:
        json.dump(backdrops_named, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved backdrops.json → {OUTPUT_BACKDROPS}")


if __name__ == "__main__":
    main()