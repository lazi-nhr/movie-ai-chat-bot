import os
import time
import re
from rdflib import Graph
from speakeasypy import Chatroom, EventType, Speakeasy


"""
To run the bot do the following:
1.  Open terminal
2.  Run "python sparql_bot.py"
3.  Wait until graph is loaded and the bot is listening (this can take a while)


To test and interact with the bot do the following:

1.  Go to https://speakeasy.ifi.uzh.ch/
2.  Login
3.  Go to "Chat"
4.  Click on "Request Chat"
5.  Enter "CyanPeekingMouse" and click on "Request"
"""

"""
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
│
└── ratings/
    ├── item_ratings.csv
    └── user_ratings.csv

"""

# a configuration dictionary facilitates administration of the code
CONFIG = {
    "Hosting": {
        "URL": "https://speakeasy.ifi.uzh.ch",
        "Username": "CyanPeekingMouse",
        "Password": "Qe5Hf3zJ"
    },
    "Data": {
        "Directory": "/space_mounts/atai-hs25/dataset",
    }
}

class Agent:
    def __init__(self):
        self.url = CONFIG["Hosting"]["URL"]
        self.username = CONFIG["Hosting"]["Username"]
        self.password = CONFIG["Hosting"]["Password"]
        self.data_dir = CONFIG["Data"]["Directory"]

        self.speakeasy = Speakeasy(host=self.url, username=self.username, password=self.password)
        self.speakeasy.login()

        self.graph = self.load_graph(self.data_dir)

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def load_graph(self, data_dir: str) -> Graph:
        g = Graph()
        nt_path = os.path.join(data_dir, "graph.nt")
        if os.path.exists(nt_path):
            print(f"Loading graph at {nt_path}...")
            g.parse(nt_path, format="nt")
            print(f"Loaded {nt_path} → {len(g)} triples")
        else:
            print(f"Graph does not exist at {nt_path}")
        return g

    def listen(self):
        self.speakeasy.start_listening()

    def on_new_message(self, message: str, room: Chatroom):
        print(f"New message in room {room.room_id}: {message}")

        try:
            ml = message.strip().lower()
            if ml.startswith("prefix") or any(k in ml for k in ("select", "ask", "construct", "describe")):
                print(f"on {room.room_id}: Detected SPARQL query, executing on local graph...")
                results = self.graph.query(message)
                reply = self.format_results(results)
                print(f"on {room.room_id}: Posting reply:\n{reply}\n")
                room.post_messages(reply)
            else:
                room.post_messages("I only process SPARQL queries at the moment.")
        except Exception as e:
            reply = f"Error processing your query: {e}"
            print(f"on {room.room_id}: {reply}")
            room.post_messages(reply)

    def on_new_reaction(self, reaction: str, message_ordinal: int, room: Chatroom):
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    def format_results(self, results) -> str:
        try:
            if results.type == 'ASK':
                return "true" if bool(results) else "false"

            if results.type == 'SELECT':
                vars_ = [str(v) for v in results.vars]
                lines = []

                for i, row in enumerate(results):
                    if i >= 50:
                        lines.append("... (truncated)")
                        break

                    if hasattr(row, "asdict"):
                        b = {str(k): v for k, v in row.asdict().items()}
                    else:
                        b = {vars_[idx]: row[idx] for idx in range(min(len(vars_), len(row)))}

                    qid = None
                    for val in b.values():
                        m = re.search(r"/entity/(Q\d+)$", str(val))
                        if m:
                            qid = m.group(1)
                            break

                    label = None
                    preferred_keys = ["directorLabel", "label", "name"] + \
                                    [k for k in b.keys() if k.lower().endswith("label")]
                    for k in preferred_keys:
                        if k in b:
                            label = str(b[k])
                            break

                    if qid and label:
                        lines.append(f"{label} ({qid})")
                    elif qid:
                        lines.append(qid)
                    else:
                        pieces = [str(b[v]) for v in vars_ if v in b]
                        lines.append(" | ".join(pieces) if pieces else str(b))

                return "\n".join(lines) if lines else "No results found."

            if results.type in ('CONSTRUCT', 'DESCRIBE'):
                g = Graph()
                for t in results:
                    g.add(t)
                preview = []
                for i, (s, p, o) in enumerate(g):
                    if i >= 10:
                        preview.append("... (truncated)")
                        break
                    preview.append(f"{s} {p} {o}")
                return f"Triples returned: {len(g)}\n" + "\n".join(preview)

            return "No results found."
        except Exception as e:
            return f"Error formatting results: {e}"

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent()
    demo_bot.listen()