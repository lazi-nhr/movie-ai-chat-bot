import json
import editdistance
import joblib
import os

import pandas as pd
import sklearn_crfsuite

from sklearn.model_selection import train_test_split

## Configuration
CONFIG = {
    "CACHE_DIR": "cache/",
    "NER": "cache/ner_dataset.csv",
    "CRF": "cache/crf_model.joblib",
    "RELATIONS_LABELS": "cache/relations/label_to_identifier.json",
    "ENTITIES_LABELS": "cache/entities/label_to_identifier.json",
}

## Install packages
#!pip -q install -U editdistance rdflib pandas numpy scikit-learn sklearn-crfsuite

class CRF():
    def __init__(self):
        self.crf_path = CONFIG["CRF"]
        self.ner_path = CONFIG["NER"]
        self.crf = self.load_crf()

    # Load named enttity recognition (NER) training data
    def load_crf(self):

        # load model if exists, otherwise train and save it
        if os.path.exists(self.crf_path):
            loaded_crf = joblib.load(self.crf_path)
            return loaded_crf

        self.df = pd.read_csv(self.ner_path, encoding = "ISO-8859-1")
        self.df = self.ds[:10000]
        self.df = self.df.ffill()
        self.sentences = self.collate(self.df)
        self.train_sentences, self.test_sentences = train_test_split(self.sentences, test_size=0.3, random_state=0)
        # X and y in the required format
        self.X_train, self.y_train = [self.sent2features(s) for s in self.train_sentences], [self.sent2labels(s) for s in self.train_sentences]
        self.X_test, self.y_test = [self.sent2features(s) for s in self.test_sentences], [self.sent2labels(s) for s in self.test_sentences]

        trained_crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',    # Limited-memory BFGS generally performs better than SGD
            c1=0.1,              # L1 regularization to encourage sparsity
            c2=0.1,              # L2 regularization to prevent overfitting
            max_iterations=1000,  # Maximum iterations
            all_possible_transitions=True,  # Include all possible state transitions
            all_possible_states=True       # Include all possible states
        )

        trained_crf.fit(self.X_train, self.y_train)
        joblib.dump(trained_crf, self.crf_path)

        return trained_crf

    @staticmethod
    def collate(dataframe):
        def to_word_pos_tag(group):
            """Convert one sentence group to a list of (word, POS, tag) tuples."""
            words = group["Word"].tolist()
            pos_tags = group["POS"].tolist()
            labels = group["Tag"].tolist()
            return list(zip(words, pos_tags, labels))

        grouped = dataframe.groupby("Sentence #").apply(to_word_pos_tag, include_groups=False)
        return list(grouped)

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'word.lower()': word.lower(),  # the word in lowercase
            'word[-3:]': word[-3:],  # last three characters
            'word[-2:]': word[-2:],  # last two characters
            'word.isupper()': word.isupper(),  # true, if the word is in uppercase
            'word.istitle()': word.istitle(),  # true, if the first character is in uppercase and remaining characters are in lowercase
            'word.isdigit()': word.isdigit(),  # true, if all characters are digits
            'postag': postag,  # POS tag
            'postag[:2]': postag[:2],  # IOB prefix
        }
        
        if i > 0:
            word1 = sent[i-1][0]  # the previous word
            postag1 = sent[i-1][1]  # POS tag of the previous word
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })  # add some features of the previous word
        else:
            features['BOS'] = True  # BOS: begining of the sentence
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]  # the next word
            postag1 = sent[i+1][1]  # POS tag of the next word
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })  # add some features of the next word
        else:
            features['EOS'] = True  # EOS: end of the sentence
        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2labels(sent):
        return [label for _, _, label in sent]



class Extraction(CRF):
    def __init__(self):
        super().__init__()
    
    # Implement NER method to extract the entity from the question
    def extract_entity(self, question):
        # Convert question into the format expected by the CRF model
        question = question[0].lower() + question[1:] # make first letter of question lowercase
        # remove any marking punctuation
        question = question.replace('?', '').replace('.', '').replace('!', '')

        words = question.split() # split into words
        
        # Assign more specific POS tags based on capitalization and position
        pos_tags = []
        for i, word in enumerate(words):
            if word.istitle() or word.isupper() or word.isdigit():
                pos_tags.append('NNP')  # Proper noun
            elif word.lower() in ['who', 'what', 'where', 'when', 'why', 'how']:
                pos_tags.append('WP')  # Wh-pronoun
            elif word.lower() in ['is', 'are', 'was', 'were']:
                pos_tags.append('VBZ')  # Verb
            elif word.lower() in ['the', 'a', 'an']:
                pos_tags.append('DT')  # Determiner
            elif word.lower() in ['of', 'in', 'by', 'with']:
                pos_tags.append('IN')  # Preposition
            else:
                pos_tags.append('NN')  # Common noun
        
        # Create sentence structure with dummy third element to match training data format
        sentence = [(word, pos, 'O') for word, pos in zip(words, pos_tags)]
        
        # Extract features
        X = [self.word2features(sentence, i) for i in range(len(sentence))]
        
        # Predict tags
        tags = self.crf.predict([X])[0]
        
        # Extract entity with more sophisticated logic
        entity = []
        in_entity = False
        
        for word, tag in zip(words, tags):
            if tag.startswith('B-'):  # Beginning of entity
                if in_entity:  # If we were already in an entity, store it and start new one
                    entity.append(' ')
                entity.append(word)
                in_entity = True
            elif tag.startswith('I-') and in_entity:  # Inside of entity
                entity.append(' ' + word)
            elif not tag.startswith('I-'):  # Outside of entity
                in_entity = False
        
        result = ''.join(entity) if entity else None
        
        # Debug information
        #print("Words:", words)
        #print("NER Tags:", tags)
        
        return result

    @staticmethod
    def link_entity(surface: str) -> tuple[str | None, float]:
        distance = 9999
        if not surface:
            return (None, distance)

        with open("cache/entities/label_to_identifier.json", "r", encoding="utf-8") as f:
            index = json.load(f)

        for key, value in index.items():
            tmp_distance = editdistance.eval(key, surface)
            if tmp_distance < distance:
                best_label, uri, distance = key, value, tmp_distance
        score = 1 - (distance / max(len(best_label), len(surface)))
        return (best_label, uri, score, distance)

    @staticmethod
    def extract_relation(question: str) -> str:
        """Extract relation from question with POS tagging"""
        # Normalize question
        question = question.strip().rstrip("?!.,").strip()
        words = question.split()
        
        # Create POS tags with focus on relation words
        pos_tags = []

        # Comprehensive list of movie-domain relation indicators
        relation_indicators = [
            # Core movie roles
            'director', 'producer', 'screenwriter', 'writer', 'actor', 'composer',
            'cast member', 'voice actor', 'narrator', 'executive producer', 'star', 
            'assistant director', 'art director', 'production designer', 'stars',
            
            # Technical roles
            'cinematographer', 'editor', 'sound designer', 'costume designer',
            'makeup artist', 'storyboard artist', 'animator', 'film editor',
            
            # Movie characteristics
            'genre', 'rating', 'classification', 'format', 'language',
            'runtime', 'release date', 'box office', 'budget',
            
            # Specific ratings
            'mpaa', 'bbfc', 'fsk', 'pegi', 'esrb',
            
            # Production related
            'studio', 'distributor', 'production company', 'filmed',
            'filmed at', 'filmed in', 'recorded at', 'shot at',
            
            # Creative elements
            'based on', 'adapted from', 'inspired by', 'music by',
            'theme by', 'score by', 'soundtrack',
            
            # Verbs (different forms)
            'directed', 'produced', 'written', 'composed',
            'edited', 'designed', 'created', 'developed'
        ]
        
        # Track potential relation words
        relation_words = []
        in_relation = False
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check if word is a relation indicator
            if word_lower in relation_indicators:
                pos_tags.append('REL')
                relation_words.append(word_lower)
                in_relation = True
            # Handle common question patterns
            elif word_lower in ['who', 'what', 'where', 'when', 'why', 'how']:
                pos_tags.append('WP')
                in_relation = False
            elif word_lower in ['is', 'are', 'was', 'were']:
                pos_tags.append('VBZ')
                in_relation = False
            elif word_lower in ['the', 'a', 'an']:
                pos_tags.append('DT')
            elif word_lower in ['of', 'in', 'by', 'with', 'for', 'at']:
                pos_tags.append('IN')
                in_relation = False
            else:
                # If we're in a relation phrase and see an unknown word, it might be part of the relation
                if in_relation and not word.istitle():  # Not likely to be part of entity if titlecased
                    pos_tags.append('REL')
                    relation_words.append(word_lower)
                else:
                    pos_tags.append('O')
                    in_relation = False
        
        # Extract relation phrase
        if relation_words:
            # Join consecutive relation words
            relation_phrase = []
            current_phrase = []
            
            for word, tag in zip(words, pos_tags):
                if tag == 'REL':
                    current_phrase.append(word.lower())
                elif current_phrase:
                    relation_phrase.append(' '.join(current_phrase))
                    current_phrase = []
                    
            if current_phrase:  # Don't forget last phrase
                relation_phrase.append(' '.join(current_phrase))
                
            # Return the longest relation phrase found
            return max(relation_phrase, key=len) if relation_phrase else None
        
        return None

    @staticmethod
    def link_relation(surface: str) -> tuple[str | None, float]:
        distance = 9999
        if not surface:
            return (None, distance)

        with open("cache/relations/label_to_identifier.json", "r", encoding="utf-8") as f:
            index = json.load(f)

        for key, value in index.items():
            tmp_distance = editdistance.eval(key, surface)
            if tmp_distance < distance:
                best_label, uri, distance = key, value, tmp_distance
        score = 1 - (distance / max(len(best_label), len(surface)))
        return (best_label, uri, score, distance)