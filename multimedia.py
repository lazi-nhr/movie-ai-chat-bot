import editdistance
import json
import os
from typing import List


class Multimedia():

    def classify_type(
        self,
        question: str
        ) -> str:

        question_lower = question.lower()

        # check for person-related multimedia keywords
        profile_keywords = [
            "picture", "image", "photo", "look like", "selfie", "portrait", "headshot"
        ]

        # check for movie-related multimedia keywords
        poster_keywords = [
            "poster", "cover", "movie poster", "film poster", "book cover"
        ]

        # check for backdrop-related multimedia keywords
        backdrop_keywords = [
            "backdrop", "scene", " scenery", "background", "setting", "landscape"
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
    def link_entity(
        surface: str,
        entity_type: str
        ) -> tuple[str | None, List[str] | None, float | None, int | None]:


        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_root = os.path.join(base_dir, "cache/multimedia")

        distance = 9999
        score = 0.0

        if not surface:
            return (None, None, None, None)
            
        if entity_type == "backdrop":
            with open(os.path.join(json_root, "backdrops.json"), "r", encoding="utf-8") as f:
                index = json.load(f)

            for key, value in index.items():
                tmp_distance = editdistance.eval(key, surface)
                if tmp_distance < distance:
                    best_label, best_values, distance = key, value, tmp_distance
            score = 1 - (distance / max(len(best_label), len(surface)))

        elif entity_type == "poster" or score < 0.5:
            with open(os.path.join(json_root, "posters.json"), "r", encoding="utf-8") as f:
                index = json.load(f)

            for key, value in index.items():
                tmp_distance = editdistance.eval(key, surface)
                if tmp_distance < distance:
                    best_label, best_values, distance = key, value, tmp_distance
            score = 1 - (distance / max(len(best_label), len(surface)))

        elif entity_type == "profile":
            with open(os.path.join(json_root, "profiles.json"), "r", encoding="utf-8") as f:
                index = json.load(f)

            for key, value in index.items():
                tmp_distance = editdistance.eval(key, surface)
                if tmp_distance < distance:
                    best_label, best_values, distance = key, value, tmp_distance
            score = 1 - (distance / max(len(best_label), len(surface)))
        
        else:
            return (None, None, None, None)

        return (best_label, best_values, score, distance)


multimedia = Multimedia()

question = "Show me a picture of Halle Berry."

entity_label = "Halle Berry"
print(f"Identified entity: {entity_label}.")

multimedia_type = multimedia.classify_type(question)
print(f"Identified multimedia type: {multimedia_type}.")

linked_label, linked_values, link_score, link_distance = multimedia.link_entity(entity_label, multimedia_type)
print(f"Linked to multimedia entity: {linked_label} with score {link_score:.2f}.")

print (linked_values)