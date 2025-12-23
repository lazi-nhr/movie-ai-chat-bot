import os
import re
import pandas as pd
import editdistance
from typing import List, Tuple, Dict, Any


class ReusableRecommendationParts:
    """Provides reusable functionality for MovieRecommender"""

    def __init__(self, u_item_path: str = None):
        self.u_item_path = u_item_path or os.path.expanduser(
            "~/.surprise_data/ml-100k/ml-100k/u.item"
        )
        self.title_to_id, self.id_to_title, self.id_to_clean_title = self.load_title_maps()
        self.metadata = None

    # -------------------------
    # Title maps / linking
    # -------------------------
    def load_title_maps(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        df = pd.read_csv(
            self.u_item_path,
            sep="|",
            header=None,
            encoding="latin-1",
            usecols=[0, 1],
            names=["movie_id", "title"]
        )

        def fix_article(title):
            match = re.match(
                r"^(?P<main>.*),\s*(?P<article>The|A|An|Le|La|Les|Il|El|Der|Die|Das|Las|Los)$",
                title,
                flags=re.IGNORECASE
            )
            if match:
                return f"{match.group('article').capitalize()} {match.group('main')}"
            return title

        df["clean_title"] = df["title"].str.replace(r"\s\(\d{4}\)$", "", regex=True)
        df["clean_title"] = df["clean_title"].apply(fix_article)
        df["movie_id"] = df["movie_id"].astype(str)

        title_to_id = dict(zip(df["clean_title"], df["movie_id"]))
        id_to_title = dict(zip(df["movie_id"], df["title"]))
        id_to_clean_title = dict(zip(df["movie_id"], df["clean_title"]))

        return title_to_id, id_to_title, id_to_clean_title

    def link_title(self, surface: str) -> Tuple[str | None, str | None]:
        if not surface or not surface.strip():
            return None, None
        surface = surface.lower().strip()

        # exact substring match first
        for title, mid in self.title_to_id.items():
            if surface in title.lower():
                return mid, title

        # fallback to edit distance
        best_dist = float("inf")
        best_mid = None
        best_title = None
        for title, mid in self.title_to_id.items():
            dist = editdistance.eval(surface, title.lower())
            if dist < best_dist:
                best_dist = dist
                best_mid = mid
                best_title = title
        if best_dist > len(surface) * 0.5:
            return None, None
        return best_mid, best_title

    # -------------------------
    # Metadata loading
    # -------------------------
    def load_metadata(self) -> Dict[str, Dict[str, Any]]:
        genre_cols = [
            "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
            "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
            "Mystery","Romance","Sci-Fi","Thriller","War","Western"
        ]
        df = pd.read_csv(
            self.u_item_path,
            sep="|",
            header=None,
            encoding="latin-1",
            usecols=[0, 1, 2] + list(range(5, 24)),
            names=["movie_id", "title", "release_date"] + genre_cols,
            low_memory=False
        )
        df["movie_id"] = df["movie_id"].astype(str)

        # extract year
        def extract_year(row):
            rd = row["release_date"]
            if isinstance(rd, str) and len(rd) >= 4:
                try:
                    return int(rd[-4:])
                except:
                    pass
            match = re.search(r"\((\d{4})\)$", row["title"])
            return int(match.group(1)) if match else None
        df["year"] = df.apply(extract_year, axis=1)

        # extract genres
        def row_genres(row):
            return [g for g in genre_cols if row.get(g, 0) == 1]
        df["genres"] = df.apply(row_genres, axis=1)

        metadata = df.set_index("movie_id")[["year", "genres"]].to_dict(orient="index")
        self.metadata = metadata
        return metadata

    # -------------------------
    # Detect clues from text
    # -------------------------
    def detect_genres(self, text: str):
        all_genres = ["biography", "romance", "action", "fantasy", "comedy", "drama",
                      "thriller", "crime", "horror", "adventure", "animation"]
        return [g.capitalize() for g in all_genres if g in text.lower()]

    def detect_language(self, text: str):
        lang_map = {
            "japanese": "Japanese",
            "french": "French",
            "german": "German",
            "spanish": "Spanish",
            "english": "English",
            "korean": "Korean"
        }
        for k, v in lang_map.items():
            if k in text.lower():
                return v
        return None

    def detect_actors(self, text: str):
        actor_list = ["Meryl Streep", "Tom Hanks", "Leonardo DiCaprio",
                      "Scarlett Johansson", "Brad Pitt"]
        return [a for a in actor_list if a.lower() in text.lower()]


# -------------------------
# MovieRecommender
# -------------------------
class MovieRecommender:
    def __init__(self, reusable: ReusableRecommendationParts):
        self.reusable = reusable
        self.metadata = self.reusable.metadata or self.reusable.load_metadata()

    def recommend(self, input_titles: List[str], top_n: int = 5) -> List[Tuple[str, str]]:
        # link input titles
        linked = [(mid, clean) for mid, clean in (self.reusable.link_title(t) for t in input_titles) if mid]
        if not linked:
            return []

        movie_ids = [mid for mid, _ in linked]
        agg_genres, agg_years = self._aggregate(movie_ids)
        scores = self._score_all_movies(agg_genres, agg_years, exclude=movie_ids)
        return self._select_top(scores, top_n)

    def recommend_from_clues(self, question: str, top_n: int = 5) -> List[Tuple[str, str]]:
        genres = self.reusable.detect_genres(question)
        language = self.reusable.detect_language(question)
        actors = self.reusable.detect_actors(question)

        candidates = []

        for mid, info in self.metadata.items():
            if genres and not set(genres).intersection(info["genres"]):
                continue
            # We can extend language or actor matching here if metadata available
            candidates.append(mid)
            if len(candidates) >= top_n:
                break

        results = [(self.reusable.id_to_clean_title[mid], mid) for mid in candidates]
        return results

    # -------------------------
    # Internal helpers
    # -------------------------
    def _aggregate(self, movie_ids: List[str]):
        genres = []
        years = []
        for mid in movie_ids:
            info = self.metadata.get(mid)
            if info:
                genres.extend(info["genres"])
                if info["year"] is not None:
                    years.append(info["year"])
        return genres, years

    def _score_all_movies(self, input_genres: List[str], input_years: List[int], exclude: List[str]):
        results = {}
        target_year = int(sum(input_years) / len(input_years)) if input_years else None

        for mid, info in self.metadata.items():
            if mid in exclude:
                continue
            movie_genres = set(info["genres"])
            genre_matches = len(set(input_genres).intersection(movie_genres))
            year_diff = abs(info["year"] - target_year) if target_year and info["year"] else 9999
            score = genre_matches - (year_diff / 10.0)
            results[mid] = {"score": score, "genres": info["genres"], "year": info["year"]}

        return results

    def _select_top(self, scores: Dict[str, Dict[str, Any]], top_n: int):
        sorted_items = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        best = sorted_items[:top_n]
        return [(self.reusable.id_to_clean_title[mid], mid) for mid, _ in best]
