"""Simple chatbot using embeddings for entity linking and prediction."""
from speakeasypy import Chatroom, EventType, Speakeasy
from . import config
from . import text_utils
from . import embeddings

class ChatBot:
    def __init__(self):
        # Initialize Speakeasy connection
        self.speakeasy = Speakeasy(
            host=config.HOSTING["URL"],
            username=config.HOSTING["Username"],
            password=config.HOSTING["Password"]
        )
        self.speakeasy.login()
        
        # Initialize embeddings
        self.embeddings = embeddings.EmbeddingIndex()
        
        # Register callbacks
        self.speakeasy.register_callback(self.on_message, EventType.MESSAGE)
        
    def start(self):
        """Start listening for messages."""
        self.speakeasy.start_listening()
        
    def on_message(self, message: str, room: Chatroom):
        """Handle incoming messages."""
        try:
            reply = self.process_message(message)
            room.post_messages(reply)
        except Exception as e:
            room.post_messages(f"Error: {str(e)}")
            
    def process_message(self, message: str) -> str:
        """Process a message and generate a response."""
        # Extract entity mention
        entity = text_utils.extract_entity(message)
        if not entity:
            return "Please mention an entity in quotes, e.g. \"The Matrix\""
            
        # Link entity
        uri, score = self.embeddings.link_entity(entity)
        if not uri or score < config.THRESHOLDS["EntityLinkMin"]:
            return f"Could not find a matching entity for \"{entity}\""
            
        # Check for relation
        relation = text_utils.extract_relation(message)
        if relation:
            # Relation prediction
            predictions = self.embeddings.predict_relation(
                uri, 
                relation,
                config.ANSWER["TopKPrediction"]
            )
            if not predictions:
                return f"Could not predict {relation} for \"{entity}\""
                
            formatted = "; ".join(
                f"{p.uri.split('/')[-1]} ({p.score:.2f})"
                for p in predictions
                if p.score >= config.THRESHOLDS["TopScoreMin"]
            )
            return f"Predicted {relation} for {entity}: {formatted}"
        
        # Default to similarity
        similar = self.embeddings.get_similar_entities(
            uri,
            config.ANSWER["TopKSimilar"]
        )
        if not similar:
            return f"Could not find similar entities for \"{entity}\""
            
        formatted = "; ".join(
            f"{s.uri.split('/')[-1]} ({s.score:.2f})"
            for s in similar
            if s.score >= config.THRESHOLDS["TopScoreMin"]
        )
        return f"Similar to {entity}: {formatted}"

if __name__ == "__main__":
    bot = ChatBot()
    bot.start()