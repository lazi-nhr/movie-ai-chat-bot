import json
import editdistance
import joblib
import os

import pandas as pd
import sklearn_crfsuite

from sklearn.model_selection import train_test_split

from config import CONFIG


class CRF():
    def __init__(self):
        self.crf_path = CONFIG["Data"]["NER"]["Model"]
        self.ner_path = CONFIG["Data"]["NER"]["Dataset"]
        self.crf = self.load_crf()

    # Load named enttity recognition (NER) training data
    def load_crf(self):

        # load model if exists, otherwise train and save it
        if os.path.exists(self.crf_path):
            loaded_crf = joblib.load(self.crf_path)
            print(f"Loaded pre-trained CRF model from {self.crf_path} for NER.")
            return loaded_crf

        self.df = pd.read_csv(self.ner_path, encoding = "ISO-8859-1")
        self.df = self.df[:10000]
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

        print(f"Trained new CRF model and saved to {self.crf_path} for NER.")
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
        self.separators = CONFIG["Format"]["Question"]["Seperators"]

    def extract_entity_simple(
            self,
            question: str
            ) -> str | None:
        """
        Extracts the substring found between single quotes (' or `) from a given question string.
        
        Args:
            question (str): The input string to search for quoted content
            
        Returns:
            str: The text found between single quotes, or empty string if no quotes found
        """
        # Find first occurrence of either quote type
        start_quote = -1
        end_quote = -1
        for i, char in enumerate(question):
            if char in self.separators:
                if start_quote == -1:
                    start_quote = i
                else:
                    end_quote = i
                break
                
        if start_quote == -1:
            return None
            
        # Extract text between quotes
        return question[start_quote + 1:end_quote-1]
    
    # Implement NER method to extract the entity from the question
    def extract_entity(
            self, 
            question: str
            ) -> str | None:
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
        
        result = ' '.join(entity) if entity else None
        
        # Debug information
        #print("Words:", words)
        #print("NER Tags:", tags)
        
        return result

    def extract_entities_rule_based(
            self,
            question: str
            ) -> list[str]:
        """
        Rule-based entity extraction specifically for movie titles.
        This complements the CRF-based approach with domain-specific patterns.
        """
        # Common trigger phrases that precede movie lists
        triggers = ['like', 'similar to', 'such as', 'including', 'e.g.', 'for example']
        
        question_lower = question.lower()
        
        # Find where the movie list starts
        start_idx = -1
        for trigger in triggers:
            idx = question_lower.find(trigger)
            if idx != -1:
                start_idx = idx + len(trigger)
                break
        
        # Also check for patterns like "Given that I like..."
        if start_idx == -1:
            if 'i like' in question_lower:
                start_idx = question_lower.find('i like') + len('i like')
        
        if start_idx == -1:
            return []
        
        # Extract the portion containing movie titles
        movie_portion = question[start_idx:].strip()
        
        # Remove trailing question words
        for ending in [', can you recommend', 'can you recommend', ', recommend', 'recommend']:
            if ending in movie_portion.lower():
                movie_portion = movie_portion[:movie_portion.lower().find(ending)]
                break
        
        # Strategy: handle two different list formats
        # Format 1: "Title1, Title2, and Title3" (with commas)
        # Format 2: "Title1 and Title2" (without commas, just "and")
        
        # Check if there are commas - if so, use comma-based splitting
        if ',' in movie_portion:
            # Replace ", and" and ", or" with just comma for easier processing
            movie_portion = movie_portion.replace(', and ', ', ').replace(', or ', ', ')
            
            # Split by commas
            raw_parts = [p.strip() for p in movie_portion.split(',')]
        else:
            # No commas - split by " and " or " or "
            # First mark the separators
            movie_portion = movie_portion.replace(' and ', '|||').replace(' or ', '|||')
            raw_parts = [p.strip() for p in movie_portion.split('|||')]
        
        # Now merge parts that should be together using smarter logic
        merged_parts = []
        i = 0
        while i < len(raw_parts):
            part = raw_parts[i].strip()
            
            # Remove leading "and"
            part_lower = part.lower()
            if part_lower.startswith('and '):
                part = part[4:].strip()
                part_lower = part.lower()
            
            # Check if this looks like a continuation
            # A part is a continuation if:
            # 1. It starts with an article followed by a single lowercase word, OR
            # 2. It starts with a lowercase word (not a complete title)
            # BUT NOT if the previous part looks complete (has capitals)
            is_continuation = False
            
            if i > 0 and merged_parts:
                # Pattern: "the Beast" after "Beauty" should merge
                # Pattern: "Pocahontas" after "Lion King" should NOT merge
                
                # Check if current part starts with article + likely continuation
                if part_lower.startswith('the ') or part_lower.startswith('a ') or part_lower.startswith('an '):
                    words_after_article = part.split()[1:] if len(part.split()) > 1 else []
                    # If there's only one word after article and it starts with capital or lowercase, likely continuation
                    if len(words_after_article) == 1:
                        is_continuation = True
                    # If multiple words and they continue a pattern (e.g., "the Beast")
                    elif len(words_after_article) > 0 and not words_after_article[0][0].isupper():
                        # Lowercase after article suggests continuation
                        is_continuation = True
            
            if is_continuation:
                # Merge with previous part
                merged_parts[-1] = merged_parts[-1] + ' and ' + part
            else:
                merged_parts.append(part)
            
            i += 1
        
        # Clean and filter
        entities = []
        for entity in merged_parts:
            entity = entity.strip().rstrip('.,;:?!')
            
            # Remove leading "and" or "or" if still present
            entity = entity.strip()
            while entity.lower().startswith('and '):
                entity = entity[4:].strip()
            while entity.lower().startswith('or '):
                entity = entity[3:].strip()
                
            if entity and len(entity) > 1:  # Avoid single letters or empty
                entities.append(entity)
        
        return entities

    def extract_entities(
            self,
            question: str
            ) -> list[str]:
        """
        Extract multiple entities from a question string.
        Combines rule-based and CRF-based approaches for better accuracy.
        
        Args:
            question (str): The input question containing multiple entities
            
        Returns:
            list[str]: A list of extracted entities
            
        Example:
            >>> extract_entities("Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween.")
            ['Nightmare on Elm Street', 'Friday the 13th', 'Halloween']
        """
        # First try rule-based approach
        rule_based_entities = self.extract_entities_rule_based(question)
        
        # Convert question into the format expected by the CRF model
        question = question[0].lower() + question[1:] # make first letter of question lowercase
        # remove any marking punctuation at the end
        question = question.rstrip('?!.')
        
        words = question.split() # split into words
        
        # Assign more specific POS tags based on capitalization and position
        pos_tags = []
        for i, word in enumerate(words):
            # Clean word from punctuation for checking but keep original for tagging
            clean_word = word.rstrip(',.;:')
            
            # Check if this might be part of a title (capitalized or following article)
            prev_word = words[i-1].lower().rstrip(',.;:') if i > 0 else ''
            next_word = words[i+1].lower().rstrip(',.;:') if i < len(words)-1 else ''
            
            if word.lower() in ['who', 'what', 'where', 'when', 'why', 'how']:
                pos_tags.append('WP')  # Wh-pronoun
            elif word.lower() in ['is', 'are', 'was', 'were', 'can', 'could', 'should', 'would']:
                pos_tags.append('VBZ')  # Verb
            elif word.lower() in ['the', 'a', 'an']:
                # Check if article is likely part of a title (followed by capitalized word)
                if i < len(words) - 1 and words[i+1].rstrip(',.;:')[0].isupper():
                    pos_tags.append('DT-TITLE')  # Determiner in title
                else:
                    pos_tags.append('DT')  # Regular determiner
            elif word.lower() in ['of', 'in', 'by', 'with', 'on', 'at', 'to', 'for', 'from']:
                # Prepositions can be part of titles
                if prev_word in ['the', 'a', 'an'] or (i > 0 and words[i-1].rstrip(',.;:')[0].isupper()):
                    pos_tags.append('IN-TITLE')  # Preposition in title
                else:
                    pos_tags.append('IN')  # Regular preposition
            elif clean_word.istitle() or clean_word.isupper() or clean_word.isdigit():
                pos_tags.append('NNP')  # Proper noun
            elif word.lower() in ['and', 'or']:
                pos_tags.append('CC')  # Coordinating conjunction
            elif word.lower() in ['like', 'similar', 'recommend', 'given', 'such']:
                pos_tags.append('JJ')  # Adjective/keyword
            else:
                # Check if lowercase word might be part of a title
                if prev_word in ['the', 'a', 'an'] or (i > 0 and words[i-1].rstrip(',.;:')[0].isupper()):
                    pos_tags.append('NN-TITLE')  # Common noun in title
                else:
                    pos_tags.append('NN')  # Common noun
        
        # Create sentence structure with dummy third element to match training data format
        sentence = [(word, pos, 'O') for word, pos in zip(words, pos_tags)]
        
        # Extract features
        X = [self.word2features(sentence, i) for i in range(len(sentence))]
        
        # Predict tags
        tags = self.crf.predict([X])[0]
        
        # Extract multiple entities with improved handling
        entities = []
        current_entity = []
        in_entity = False
        
        for i, (word, tag, pos) in enumerate(zip(words, tags, pos_tags)):
            # Clean word from trailing punctuation for entity extraction
            clean_word = word.rstrip(',.;:')
            word_lower = clean_word.lower()
            
            # Skip connector words between entities
            if word_lower in ['and', 'or', ','] and not in_entity:
                continue
                
            if tag.startswith('B-'):  # Beginning of entity
                if current_entity:  # Store previous entity if exists
                    entity_text = ' '.join(current_entity).strip()
                    if entity_text and entity_text.lower() not in ['i', 'like', 'given', 'that']:
                        entities.append(entity_text)
                current_entity = [clean_word]
                in_entity = True
            elif tag.startswith('I-') and in_entity:  # Inside of entity
                current_entity.append(clean_word)
            elif in_entity:
                # Check if this is an article or preposition that might be part of a title
                if word_lower in ['the', 'a', 'an', 'of', 'on', 'in', 'and']:
                    # Look ahead to see if there's more capitalized words
                    if i < len(words) - 1:
                        next_word = words[i+1].rstrip(',.;:')
                        if next_word and next_word[0].isupper():
                            # This article/preposition is likely part of the title
                            current_entity.append(clean_word)
                            continue
                
                # End of entity
                if current_entity:
                    entity_text = ' '.join(current_entity).strip()
                    # Filter out common false positives
                    if entity_text and entity_text.lower() not in ['i', 'like', 'given', 'that', 'and', 'or']:
                        entities.append(entity_text)
                    current_entity = []
                in_entity = False
            else:  # Outside of entity
                # Check if this is a capitalized word that might start an entity
                # This catches cases where the CRF missed an entity
                if clean_word and clean_word[0].isupper() and word_lower not in ['i']:
                    # Look for sequences of capitalized words
                    if i < len(words) - 1:
                        next_word = words[i+1].rstrip(',.;:')
                        if next_word and (next_word[0].isupper() or next_word.lower() in ['the', 'of', 'on', 'in', 'and']):
                            current_entity = [clean_word]
                            in_entity = True
        
        # Don't forget the last entity if sentence ends with one
        if current_entity and in_entity:
            entity_text = ' '.join(current_entity).strip()
            if entity_text and entity_text.lower() not in ['i', 'like', 'given', 'that', 'and', 'or']:
                entities.append(entity_text)
        
        # Debug information
        #print("Words:", words)
        #print("NER Tags:", tags)
        #print("Extracted entities before refinement:", entities)
        
        # If rule-based extraction worked and looks good, prefer it
        if rule_based_entities and len(rule_based_entities) > 0:
            # Validate each entity through linking
            validated_entities = []
            for entity in rule_based_entities:
                linked_label, uri, score, distance = self.link_entity(entity)
                # If we found a good match, use the canonical label; otherwise keep original
                if linked_label and score and score > 0.6:
                    validated_entities.append(linked_label)
                else:
                    validated_entities.append(entity)
            
            # If validation was successful, use rule-based results
            if validated_entities:
                #print("Using rule-based extraction:", validated_entities)
                return validated_entities
        
        # Otherwise, post-process CRF entities: try to improve entities using entity linking
        refined_entities = []
        for entity in entities:
            # Try linking to see if we can find a better match
            linked_label, uri, score, distance = self.link_entity(entity)
            
            # If we found a very close match, use the canonical label
            if linked_label and score and score > 0.8:
                refined_entities.append(linked_label)
            else:
                # Keep original extraction
                refined_entities.append(entity)
        
        #print("Refined entities:", refined_entities)
        return refined_entities

    @staticmethod
    def link_entity(
        surface: str
        ) -> tuple[str | None, str | None, float | None, int | None]:

        distance = 9999
        if not surface:
            return (None, None, None, None)

        with open("cache/entities/label_to_identifier.json", "r", encoding="utf-8") as f:
            index = json.load(f)

        for key, value in index.items():
            tmp_distance = editdistance.eval(key, surface)
            if tmp_distance < distance:
                best_label, uri, distance = key, value, tmp_distance
        score = 1 - (distance / max(len(best_label), len(surface)))
        return (best_label, uri, score, distance)

    @staticmethod
    def extract_relation(
        question: str,
        entity_label: str | None = None
        ) -> str | None:
        """
        Extract relation from question with specific handling for film-related queries.
        Handles both direct mentions and question-based implications.
        
        Args:
            question (str): The question to extract relation from
            entity_label (str | None): Optional entity label to remove from question
            
        Returns:
            str | None: Extracted relation or None if no relation found
        """
        # Load film-specific relations
        with open("cache/relations/film_relations.json", "r", encoding="utf-8") as f:
            film_relations = json.load(f)
            
        # Normalize question
        question = question.lower().strip().rstrip("?!.,'`\"").strip()
        
        # Remove entity_label from question if provided
        if entity_label:
            question = question.replace(entity_label.lower(), "").strip()
        
        # Direct pattern matching for common film relation phrases
        for relation_phrase, _ in film_relations.items():
            if relation_phrase in question:
                return relation_phrase
                
        # Question-based relation mapping
        words = question.split()
        
        # Look for question patterns that imply specific relations
        if "who" in words:
            if any(w in words for w in ["direct", "directed", "director"]):
                return "director"
            if any(w in words for w in ["write", "wrote", "written", "writer", "screenwriter"]):
                return "screenwriter"
            if any(w in words for w in ["act", "acted", "actor", "star", "starring"]):
                return "cast member"
            if any(w in words for w in ["composer", "music", "score", "soundtrack", "composed", "compose"]):
                return "composer"
                
        if "what" in words:
            if "genre" in words or "type" in question:
                return "genre"
            if any(w in words for w in ["country", "from"]):
                return "country of origin"
            if any(w in words for w in ["award", "awards"]):
                return "award received"
            if any(w in words for w in ["nominated", "nomination"]):
                return "nominated for"
                
        if "when" in words:
            if any(phrase in question for phrase in ["come out", "came out", "release", "released"]):
                return "publication date"
                
        # Handle special cases where the relation might be implied
        if "genre" in question or "type of film" in question or "kind of movie" in question:
            return "genre"
        if "country" in question:
            return "country of origin"
            
        # If no specific relation is found, return None
        return None

    @staticmethod
    def link_relation(
        surface: str
        ) -> tuple[str | None, str | None, float | None, int | None]:
        
        distance = 9999
        if not surface:
            return (None, None, None, None)

        with open("cache/relations/film_relations.json", "r", encoding="utf-8") as f:
            index = json.load(f)

        for key, value in index.items():
            tmp_distance = editdistance.eval(key, surface)
            if tmp_distance < distance:
                best_label, uri, distance = key, value, tmp_distance
        score = 1 - (distance / max(len(best_label), len(surface)))
        return (best_label, uri, score, distance)