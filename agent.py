import re
import json
from speakeasypy import Chatroom, EventType, Speakeasy
from extraction import Extraction
from embeddings import Embeddings
from factual import Factual

"""
To run the bot do the following:
1.  Open terminal
2.  Run "python agent.py"
3.  Wait until graph is loaded and the bot is listening (this can take a while)


To test and interact with the bot do the following:

1.  Go to https://speakeasy.ifi.uzh.ch/
2.  Login
3.  Go to "Chat"
4.  Click on "Request Chat"
5.  Enter "CyanPeekingMouse" and click on "Request"
"""

# a configuration dictionary facilitates administration of the code
with open("config.json", "r") as f:
    CONFIG = json.load(f)

class Agent:
    def __init__(self):
        self.url = CONFIG["Hosting"]["URL"]
        self.username = CONFIG["Hosting"]["Username"]
        self.password = CONFIG["Hosting"]["Password"]

        self.extraction = Extraction()
        self.embeddings = Embeddings()
        self.factual = Factual()

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        print(f"New message in room {room.room_id}: {message}")

        try:
            reply = self.process_question(message)
            room.post_messages(reply)
        except Exception as e:
            reply = f"Error processing your query: {e}"
            print(f"on {room.room_id}: {reply}")
            room.post_messages(reply)

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom):
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    @staticmethod
    def classify_question(question: str) -> tuple[str, str]:
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
    
    @staticmethod
    def classify_question_simple(question: str) -> tuple[str, str]:
        """
        A simpler version of question classification that looks for keywords and splits on ':'
        
        Args:
            question (str): The input question
            
        Returns:
            tuple[str, str]: (pure_question, question_type)
            - pure_question: The question after the first colon
            - question_type: One of 'general', 'factual', or 'embedding'
        """
        question = question.lower()
        
        # Determine type based on keywords
        if "factual" in question:
            question_type = "factual"
        elif "embedding" in question:
            question_type = "embedding"
        else:
            question_type = "general"
            
        # Extract pure question after the first colon
        if ":" in question:
            pure_question = question.split(":", 1)[1].strip()
        else:
            pure_question = question
            
        return pure_question, question_type
    
    def process_question(self, question: str) -> str:
        pure_q, q_type = self.classify_question(question)

        entity = self.extraction.extract_entity(pure_q)
        entity_label, entity_uri, entity_score, entity_distance = self.extraction.link_entity(entity)
        
        relation = self.extraction.extract_relation(pure_q)
        relation_label, relation_uri, relation_score, relation_distance = self.extraction.link_relation(relation)

        if q_type == "factual" or q_type == "general":
            # sparql_query = self.factual.translate_to_sparql(entity_uri, relation_uri)
            # results = self.factual.sparql_query(sparql_query) # this should return a list with entities
            # example format of an answer: 
            # "The factual answer is: Ethan Coen and Joel Coen"
            # "The factual answer is: drama film and biographical film and crime film"

            dummy_results = ["Ethan Coen", "Joel Coen"]  # Placeholder for actual results
            result = " and ".join(dummy_results)
            return f"The factual answer is: {result}"
        elif q_type == "embedding":

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


if __name__ == '__main__':
    demo_bot = Agent()
    demo_bot.listen()