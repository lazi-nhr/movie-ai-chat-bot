#!/usr/bin/env python3

import os
import json
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from imdb import IMDb

# ============================================================
# CONFIGURATION
# Get imdb data from https://datasets.imdbws.com/
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(base_dir, "dataset")
IMAGES_JSON = os.path.join(DATA_ROOT, "additional", "images.json")

# IMDb TSV files (adjust if stored elsewhere)
NAME_TSV = os.path.join(base_dir, "cache/multimedia/name.basics.tsv.gz")
TITLE_TSV = os.path.join(base_dir, "cache/multimedia/title.basics.tsv.gz")

# Output directory
OUT_DIR = "./cache/multimedia"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PROFILES  = os.path.join(OUT_DIR, "profiles.json")
OUT_POSTERS   = os.path.join(OUT_DIR, "posters.json")
OUT_BACKDROPS = os.path.join(OUT_DIR, "backdrops.json")

REQUEST_SLEEP = 0.1  # fallback IMDbPY delay (rare)


# ============================================================
# HELPERS
# ============================================================

def normalize_image_id(img):
    """Remove .jpg extension."""
    return img[:-4] if img.lower().endswith(".jpg") else img


def load_name_lookup(path):
    """
    Loads nconst -> primaryName for *only the required columns*.
    """
    print("[INFO] Loading people lookup (nconst -> primaryName)")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["nconst", "primaryName"]
    )
    return dict(zip(df["nconst"], df["primaryName"]))


def load_title_lookup(path):
    """
    Loads tconst -> primaryTitle for movies.
    """
    print("[INFO] Loading movie lookup (tconst -> primaryTitle)")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["tconst", "titleType", "primaryTitle"]
    )

    # keep movies only
    df = df[df["titleType"] == "movie"]

    return dict(zip(df["tconst"], df["primaryTitle"]))


def fallback_imdb_lookup_person(pid, ia):
    code = pid[2:] if pid.startswith("nm") else pid
    try:
        person = ia.get_person(code)
        if person:
            return person.get("name")
    except Exception:
        return None
    return None


def fallback_imdb_lookup_movie(mid, ia):
    code = mid[2:] if mid.startswith("tt") else mid
    try:
        movie = ia.get_movie(code)
        if movie:
            return movie.get("title")
    except Exception:
        return None
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(IMAGES_JSON):
        raise FileNotFoundError(IMAGES_JSON)

    print("[INFO] Reading images.json ...")
    with open(IMAGES_JSON, "r", encoding="utf-8") as f:
        images = json.load(f)

    profiles_by_pid = defaultdict(list)
    posters_by_mid = defaultdict(list)
    backdrops_by_mid = defaultdict(list)

    print("[INFO] Extracting IDs from images.json")
    for item in tqdm(images, unit="img"):
        img = item.get("img")
        t = item.get("type")
        movies = item.get("movie", [])
        people = item.get("cast", [])

        if not img or not t:
            continue

        img_id = normalize_image_id(img)

        if t == "profile":
            for pid in people:
                profiles_by_pid[pid].append(img_id)

        elif t == "poster":
            for mid in movies:
                posters_by_mid[mid].append(img_id)

        elif t == "backdrop":
            for mid in movies:
                backdrops_by_mid[mid].append(img_id)

    # ============================================================
    # Load T-SV maps
    # ============================================================

    print("[INFO] Loading local IMDb TSV lookup tables")
    people_map = load_name_lookup(NAME_TSV)
    movie_map  = load_title_lookup(TITLE_TSV)

    # IMDbPY fallback
    ia = IMDb()

    # ============================================================
    # Build final name mappings
    # ============================================================

    profiles_named = defaultdict(list)
    posters_named = defaultdict(list)
    backdrops_named = defaultdict(list)

    print("[INFO] Resolving PERSON names")
    for pid, imgs in tqdm(profiles_by_pid.items(), unit="pid"):
        name = people_map.get(pid)
        if not name:  # fallback only if missing
            name = fallback_imdb_lookup_person(pid, ia)
            if not name:
                name = pid  # last fallback
            time.sleep(REQUEST_SLEEP)
        profiles_named[name].extend(imgs)

    print("[INFO] Resolving MOVIE titles for posters")
    for mid, imgs in tqdm(posters_by_mid.items(), unit="mid"):
        title = movie_map.get(mid)
        if not title:
            title = fallback_imdb_lookup_movie(mid, ia)
            if not title:
                title = mid
            time.sleep(REQUEST_SLEEP)
        posters_named[title].extend(imgs)

    print("[INFO] Resolving MOVIE titles for backdrops")
    for mid, imgs in tqdm(backdrops_by_mid.items(), unit="mid"):
        title = movie_map.get(mid)
        if not title:
            title = fallback_imdb_lookup_movie(mid, ia)
            if not title:
                title = mid
            time.sleep(REQUEST_SLEEP)
        backdrops_named[title].extend(imgs)

    # ============================================================
    # Write JSON outputs
    # ============================================================

    with open(OUT_PROFILES, "w", encoding="utf-8") as f:
        json.dump(dict(profiles_named), f, indent=2, ensure_ascii=False)

    with open(OUT_POSTERS, "w", encoding="utf-8") as f:
        json.dump(dict(posters_named), f, indent=2, ensure_ascii=False)

    with open(OUT_BACKDROPS, "w", encoding="utf-8") as f:
        json.dump(dict(backdrops_named), f, indent=2, ensure_ascii=False)

    print("\n[INFO] Done!")
    print(f"Profiles saved to:  {OUT_PROFILES}")
    print(f"Posters saved to:   {OUT_POSTERS}")
    print(f"Backdrops saved to: {OUT_BACKDROPS}")


if __name__ == "__main__":
    main()