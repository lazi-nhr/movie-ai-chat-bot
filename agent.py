import re
import json
from random import choice

from speakeasypy import Chatroom, EventType, Speakeasy
from extraction import Extraction
from embeddings import Embeddings
from factual import Factual
from recommendation import Recommendation
from multimedia import Multimedia
from config import CONFIG


"""
To install all packages run this in the Terminal once:
pip install "numpy<2" scikit-surprise
pip install pandas editdistance rdflib joblib scikit-learn sklearn-crfsuite

To run the bot do the following:
1.  Open terminal
2.  Enter "conda activate recommendation" to run on python 3.11.14 which is necessary for the suprise library
2.  Enter "cd Project Submission 3/code"
3.  Run "python agent.py"
4.  Wait until graph is loaded and the bot is listening (this can take a while)


To test and interact with the bot do the following:

1.  Go to https://speakeasy.ifi.uzh.ch/
2.  Login
3.  Go to "Chat"
4.  Click on "Request Chat"
5.  Enter "CyanPeekingMouse" and click on "Request"

The data is strucutred as follows:

The dataset directory structure is as follows:

/space_mounts/atai-hs25/dataset/
├── additional/
│   ├── images.json
│   ├── movie_plots.csv
│   ├── plots.csv
│   └── user_comments.csv
│
├── embeddings/
│   ├── entity_embeds.npy
│   ├── entity_ids.del
│   ├── relation_embeds.npy
│   └── relation_ids.del
│
├── graph.nt
├── graph.tsv
│
├── image_features/
│   ├── 0000.pkl
│   ├── 0001.pkl
│   ├──  ... 
│   └── 0347.pkl
│
├── images/
│   ├── 0000/
│   ├── 0001/
│   ├──  ... 
│   └── 0347/
│       ├── 15f1zhXXmhADnscuAdDayGNdXpk.jpg
│       ├── 17U6qlr2QR79MVypisjRD9CtKtg.jpg
│       ├──  ...
│       └── zZnaTUuf3TmOq6seZAHQmU7BLaE.jpg
│
└── ratings/
    ├── item_ratings.csv
    └── user_ratings.csv
"""

class Agent:
    def __init__(self):
        self.url = CONFIG["Hosting"]["URL"]
        self.username = CONFIG["Hosting"]["Username"]
        self.password = CONFIG["Hosting"]["Password"]

        self.extraction = Extraction()
        self.embeddings = Embeddings()
        self.factual = Factual()
        self.recommendation = Recommendation()
        self.multimedia = Multimedia()

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        # print(f"New message in room {room.room_id}: {message}")
        print("\n")
        print(f"New message: {message}")

        try:
            reply = self.process_question(message)
            print(f"Final reply: {reply}")
            room.post_messages(reply)
        except Exception as e:
            reply = f"Error processing your query: {e}"
            print(f"on {room.room_id}: {reply}")
            room.post_messages(reply)
        
        print("\n")

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom):
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    @staticmethod
    def classify_question_ev2(question: str) -> tuple[str, str]:
        """
        Classifies a question into one of three types (general, factual, or embedding)
        and extracts the pure question without the format prefix.
        
        Args:
            question (str): The input question with format prefix
            
        Returns:
            tuple[str, str]: (pure_question, question_type)
            - pure_question: The question without the format prefix
            - question_type: One of 'general', 'factual', or 'embedding'
        """
        def format_to_regex(fmt: str) -> re.Pattern:
            # Escape the format string but keep the {question} placeholder
            escaped = re.escape(fmt)
            # Replace the escaped {question} with a capturing group
            pattern = escaped.replace(r'\{question\}', r'(.+)')
            return re.compile(f"^{pattern}$")
        
        # Create patterns from CONFIG formats
        patterns = {
            question_type.lower(): format_to_regex(fmt)
            for question_type, fmt in CONFIG["Format"]["Question"].items()
        }
        
        # Try to match each pattern
        for question_type, pattern in patterns.items():
            match = pattern.match(question)
            if match:
                return match.group(1), question_type
                
        # If no pattern matches, assume it's a general question
        return question, "general"
    
    def classify_question(self, question: str) -> tuple[str, str]:
        """
        A simpler version of question classification that looks for keywords and splits on ':'
        
        Args:
            question (str): The input question
            
        Returns:
            tuple[str, str]: (pure_question, question_type)
            - pure_question: The question after the first colon
            - question_type: One of 'general', 'factual', or 'embedding'
        """
        question_lower = question.lower()
        
        # check for SPARQL query indicators
        if question_lower.lstrip().startswith("prefix") or re.search(r"\bselect\b.*\bwhere\b", question_lower):
            return question, "sparql"
        
        # check for keywords
        # recommendation keywords
        elif any(k in question_lower for k in ("recommend", "suggest")):
            question_type = "recommendation"

        # factual keywords
        elif "factual" in question_lower:
            question_type = "factual"

        # embedding keywords
        elif "embedding" in question_lower:
            question_type = "embedding"

        # default to general
        else:
            # Let the Multimedia module decide if this is multimedia,
            # otherwise default to general.
            multimedia_type = self.multimedia.classify_type(question)
            if multimedia_type is not None:
                question_type = "multimedia"
            else:
                question_type = "general"
            
        # Extract pure question after the first colon (NOT NECESSARY ANYMORE)
        #if ":" in question:
        #    pure_question = question.split(":", 1)[1].strip()
        #else:
        #    pure_question = question
            
        return question, question_type
    
    def process_question(self, question: str) -> str:
        pure_q, q_type = self.classify_question(question)
        print(f"Classified  as type '{q_type}'.")
        
        relation = self.extraction.extract_relation(pure_q)
        relation_label, relation_uri, relation_score, relation_distance = self.extraction.link_relation(relation)

        if q_type == "factual" or q_type == "general":
            entity = self.extraction.extract_entity_simple(pure_q) # simplified entity extraction for factual/general
            entity_label, entity_uri, entity_score, entity_distance = self.extraction.link_entity(entity)
            print(f"Identified entity: '{entity_label}'.")
            print(f"Identified relation: '{relation_label}'.")
        
            sparql_query = self.factual.translate_to_sparql(entity_uri, relation_uri)
            results = self.factual.sparql_query(sparql_query) # this should return a list with entities
            formatted_results = self.factual.get_labels(results)
            if formatted_results != "No results found.":
                return f"{formatted_results}"
            else:
                print("No results using SPARQL.")
                print("Falling back to embedding-based retrieval.")
                results = self.embeddings.get_best_result(
                    entity_uri,
                    relation_uri,
                    1
                )
                head_uri, head_label, head_score, head_rank = results[0]

                with open("cache/entities/entity_types.json", "r", encoding="utf-8") as f:
                    entity_types = json.load(f)
                
                type_qid = entity_types.get(head_uri, "N/A")
                return f"I think {head_label}"
    
        elif q_type == "sparql":
            results = self.factual.sparql_query(pure_q)
            formatted_results = self.factual.get_labels(results)
            return f"The result is: {formatted_results}"

        elif q_type == "recommendation":
            movie_list = self.extraction.extract_entities(pure_q)
            print(f"Identified movies: '{movie_list}'.")
            movies = self.recommendation.recommend_from_titles(movie_list)
            filtered_movies = self.recommendation.filter_recommendations(movie_list, movies["recommendations"])
            recs = []
            for recommendation in filtered_movies:
                mid = recommendation["movie_id"]
                recs.append(self.recommendation.id_to_clean_title[mid])
                #recs.append(recommendation["title"])
            return " and ".join(recs) if recs else "No results found."
        
        elif q_type == "embedding":
            entity = self.extraction.extract_entity(pure_q)
            entity_label, entity_uri, entity_score, entity_distance = self.extraction.link_entity(entity)
            print(f"Identified entity: '{entity_label}'.")
            print(f"Identified relation: '{relation_label}'.")

            results = self.embeddings.get_best_result(
                entity_uri,
                relation_uri,
                1
            )
            head_uri, head_label, head_score, head_rank = results[0]

            with open("cache/entities/entity_types.json", "r", encoding="utf-8") as f:
                entity_types = json.load(f)
            
            type_qid = entity_types.get(head_uri, "N/A")
            return f"The answer suggested by embeddings is: {head_label} (type: {type_qid})"
        
        elif q_type == "multimedia":
            entity = self.extraction.extract_entity(pure_q)
            entity_label, entity_uri, entity_score, entity_distance = self.extraction.link_entity(entity)
            print(f"Extracted entity: '{entity_label}'.")
            multimedia_type = self.multimedia.classify_type(pure_q)
            print(f"Identified multimedia type: '{multimedia_type}'.")
            linked_label, linked_values, link_score, link_distance = self.multimedia.link_entity(entity_label, multimedia_type)
            print(f"Linked to multimedia entity: '{linked_label}' with score {link_score:.2f}.")

            # randomly select one multimedia value to return
            if linked_values:
                selected_value = choice(linked_values)
                
                return f"image:{selected_value}"
            else:
                return f"No multimedia found for {linked_label}."



if __name__ == '__main__':
    demo_bot = Agent()
    demo_bot.listen()