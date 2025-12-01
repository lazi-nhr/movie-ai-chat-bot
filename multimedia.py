import editdistance
import json
import os
from typing import List, Tuple, Optional


class Multimedia:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(base_dir, "cache/multimedia")

        with open(os.path.join(root, "profiles.json"), "r", encoding="utf-8") as f:
            self.profiles_index = json.load(f)
        with open(os.path.join(root, "posters.json"), "r", encoding="utf-8") as f:
            self.posters_index = json.load(f)
        with open(os.path.join(root, "backdrops.json"), "r", encoding="utf-8") as f:
            self.backdrops_index = json.load(f)

    def classify_type(self, question: str) -> str:
        question_lower = question.lower()

        profile_keywords = [
            "picture", "image", "photo", "look like", "selfie", "portrait", "headshot"
        ]
        poster_keywords = [
            "poster", "cover", "movie poster", "film poster", "book cover"
        ]
        backdrop_keywords = [
            "backdrop", "scene", "scenery", "background", "setting", "landscape"
        ]
        
        if any(keyword in question_lower for keyword in profile_keywords):
            return "profile"
        elif any(keyword in question_lower for keyword in poster_keywords):
            return "poster"
        elif any(keyword in question_lower for keyword in backdrop_keywords):
            return "backdrop"
        else:
            return "profile"  # default to profile if unsure

    @staticmethod
    def _best_match(
        surface: str,
        index: dict[str, List[str]]
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[float], Optional[int]]:

        if not surface or not index:
            return None, None, None, None

        surface_norm = surface.lower()
        best_label: Optional[str] = None
        best_values: Optional[List[str]] = None
        best_distance: Optional[int] = None

        for key, value in index.items():
            key_norm = key.lower()
            d = editdistance.eval(key_norm, surface_norm)
            if best_distance is None or d < best_distance:
                best_label = key          # keep original casing for output
                best_values = value
                best_distance = d

        if best_label is None or best_distance is None:
            return None, None, None, None

        score = 1 - (best_distance / max(len(best_label), len(surface)))
        return best_label, best_values, score, best_distance

    def link_entity(
        self,
        surface: str,
        entity_type: str
    ) -> tuple[str | None, List[str] | None, float | None, int | None]:

        if not surface:
            return (None, None, None, None)

        # pick the right index once
        index_map = {
            "profile":  self.profiles_index,
            "poster":   self.posters_index,
            "backdrop": self.backdrops_index,
        }

        index = index_map.get(entity_type)
        if index is None:
            return (None, None, None, None)

        return self._best_match(surface, index)