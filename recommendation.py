import json
import os
import re
import numpy as np
import pandas as pd

import editdistance

from surprise import SVD
from surprise import Dataset
from surprise.model_selection.split import train_test_split

from sklearn.metrics.pairwise import cosine_similarity


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

class Recommendation():

    def __init__(self):
        self.svd_algo, self.trainset = self.load_svd()
        self.title_to_id, self.id_to_title = self.load_title_maps()

    def load_svd(self):
        data = Dataset.load_builtin('ml-100k')
        # format: (user id, movie id, rating)
        trainset, _ = train_test_split(data, test_size=.01, random_state=42) # no need for test (.0)
        svd_algo = SVD(random_state = 42)
        svd_algo.fit(trainset)
        return svd_algo, trainset
    
    def load_title_maps(self, u_item_path=None):
        # default path Surprise uses for ml-100k cache
        default_path = os.path.expanduser("~/.surprise_data/ml-100k/ml-100k/u.item")
        path = u_item_path or default_path

        # read id|title from u.item
        df = pd.read_csv(
            path, sep='|', header=None, encoding='latin-1',
            usecols=[0, 1], names=['movie_id', 'title']
        )

        # reformat "Title, The" -> "The Title"
        def fix_article(title):
            match = re.match(r'^(?P<main>.*),\s*(?P<article>The|A|An|Le|La|Les|Il|El|Der|Die|Das|Las|Los)$', title, flags=re.IGNORECASE)
            if match:
                article = match.group('article').capitalize()
                main = match.group('main')
                return f"{article} {main}"
            return title

        # remove trailing year pattern
        df['clean_title'] = df['title'].str.replace(r'\s\(\d{4}\)$', '', regex=True)
        # fix articles
        df['clean_title'] = df['clean_title'].apply(fix_article)

        # convert ids to string for Surprise compatibility
        df['movie_id'] = df['movie_id'].astype(str)

        title_to_id = dict(zip(df['clean_title'], df['movie_id']))
        id_to_title = dict(zip(df['movie_id'], df['title']))
        return title_to_id, id_to_title
    

    def link_title(self, surface):
        """
        surface: user-provided movie title (string)
        title_to_id: dict mapping official titles to raw ids
        """
        surface = surface.lower()
        # partial match
        for title, id in self.title_to_id.items():
            if surface in title.lower():
                return id, title

        # lowest edit distance match
        best_title = None
        best_id = None
        best_dist = 9999
        for title, id in self.title_to_id.items():
            if surface in title.lower():
                return id, title
            dist = editdistance.eval(surface, title.lower())
            if dist < best_dist:
                best_dist = dist
                best_title = title
                best_id = id
        if best_dist > len(surface) * 0.5:  # reject if >50% different
            return None, None
        return best_id, best_title
    

    def recommend_from_titles(
        self,
        titles,
        top_n=10,
        per_item_pool=500,
        exclude_input=True
    ):
        """
        titles: list of movie titles (strings)
        svd_algo: fitted Surprise SVD
        trainset: Surprise Trainset returned by your load_svd()
        per_item_pool: for each liked title, how many nearest neighbors to pool before aggregating
        """
        # get item-factor matrix (inner-item-id order)
        Q = self.svd_algo.qi  # shape: [n_items, n_factors]
        n_items = Q.shape[0]

        # map input titles -> inner ids
        liked_inner_ids = []
        missing = []
        for t in titles:
            raw_iid, t = self.link_title(t)
            if raw_iid is None:
                missing.append(t)
                continue
            try:
                inner_id = self.trainset.to_inner_iid(raw_iid)
                liked_inner_ids.append(inner_id)
            except ValueError:
                # item not in trainset (rare in ML-100k if using full data)
                missing.append(t)

        if not liked_inner_ids:
            raise ValueError(
                f"None of the {len(titles)} provided titles could be mapped to the dataset.\n"
                f"Attempted: {titles}\n"
                f"Not found: {missing}"
            )

        # compute aggregated similarity scores
        agg_scores = np.zeros(n_items, dtype=np.float64)
        for inner_id in liked_inner_ids:
            v = Q[inner_id].reshape(1, -1)
            sims = cosine_similarity(v, Q).ravel()
            sims[inner_id] = 0.0  # don’t recommend the exact same item
            # optionally keep only the strongest neighbors per seed to reduce noise
            if per_item_pool and per_item_pool < n_items:
                top_idx = np.argpartition(-sims, per_item_pool)[:per_item_pool]
                pruned = np.zeros_like(sims)
                pruned[top_idx] = sims[top_idx]
                sims = pruned
            agg_scores += sims

        # exclude the input movies, if desired
        if exclude_input:
            agg_scores[liked_inner_ids] = -np.inf

        # rank and build output
        top_inner = np.argsort(-agg_scores)[:top_n]
        recs = []
        for iid in top_inner:
            raw_iid = self.trainset.to_raw_iid(iid)
            title = self.id_to_title.get(raw_iid, f"[item {raw_iid}]")
            score = agg_scores[iid]
            recs.append({"movie_id": raw_iid, "title": title, "score": float(score)})

        return {
            "missing_titles": missing,   # titles we couldn’t map
            "recommendations": recs      # ranked list
        }


# --- usage example ---
rec = Recommendation()
movies = ["Nightmare on Elm Street", "Friday the 13th", "Halloween"]
result = rec.recommend_from_titles(movies, top_n=3)
print(result)