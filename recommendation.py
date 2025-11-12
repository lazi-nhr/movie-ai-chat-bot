import os
import re
import numpy as np
import pandas as pd

import editdistance

from surprise import SVD
from surprise import Dataset
from surprise.model_selection.split import train_test_split

from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG


# virtual environment necessary: python 3.11.14
class Recommendation():

    def __init__(self):
        self.svd_algo, self.trainset = self.load_svd() # train and load SVD model
        self.title_to_id, self.id_to_title = self.load_title_maps() # load title mappings

    def load_svd(self):
        data = Dataset.load_builtin('ml-100k') # load MovieLens 100k dataset
        # format: (user id, movie id, rating)
        trainset, _ = train_test_split(data, test_size=.01, random_state=42)
        svd_algo = SVD(random_state = 42)
        svd_algo.fit(trainset) # train
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
    
    # add filter: include only same genres, year range +- 5 years
    def recommend_from_titles(
        self,
        titles,
        top_n=10,
        per_item_pool=500
    ):
        """
        titles: list of movie titles (strings)
        svd_algo: fitted Surprise SVD
        trainset: Surprise Trainset returned by your load_svd()
        per_item_pool: for each liked title, how many nearest neighbors to pool before aggregating
        """
        # get item-factor matrix (inner-item-id order)
        Q = self.svd_algo.qi  # shape: [n_items, n_factors]
        n_items = Q.shape[0] # total number of movies

        # map input titles -> inner ids
        liked_inner_ids = []
        missing = []
        for title in titles:
            raw_iid, t = self.link_title(title)
            if raw_iid is None:
                missing.append(title)
                continue
            try:
                inner_id = self.trainset.to_inner_iid(raw_iid) # get inner id
                liked_inner_ids.append(inner_id) # collect inner ids
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
            v = Q[inner_id].reshape(1, -1) # get row of movie, reshape to 2D array -> extract latenet vector, 
            sims = cosine_similarity(v, Q).ravel() # compute cosine similarity against all items
            sims[inner_id] = 0.0  # don’t recommend the exact same item (self-similarity = 0)
            # optionally keep only the strongest neighbors per seed to reduce noise
            if per_item_pool and per_item_pool < n_items: # prune weak similarites if enabled
                top_idx = np.argpartition(-sims, per_item_pool)[:per_item_pool] # get top movie indices by similarity
                pruned = np.zeros_like(sims) # array of zeros, same size as sims
                pruned[top_idx] = sims[top_idx] # keep only top similarities
                sims = pruned # replace sims with pruned version
            agg_scores += sims # aggregate scores for all liked items

        # exclude the input movies
        agg_scores[liked_inner_ids] = -np.inf

        # rank and build output
        top_inner = np.argsort(-agg_scores)[:top_n] # get top-n inner ids by descending order (-)
        recs = []
        for iid in top_inner:
            raw_iid = self.trainset.to_raw_iid(iid) # convert back to raw id
            title = self.id_to_title.get(raw_iid, f"[item {raw_iid}]") # get title
            score = agg_scores[iid] # get score
            recs.append({"movie_id": raw_iid, "title": title, "score": float(score)})

        return {
            "missing_titles": missing,   # titles we couldn’t map
            "recommendations": recs      # ranked list
        }


# --- usage example ---
rec = Recommendation()
movies = ["The Lion King", "Pocahontas", "The Beauty and the Beast"]
result = rec.recommend_from_titles(movies, top_n=3)
print(result)