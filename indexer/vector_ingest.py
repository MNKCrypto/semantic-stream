import os
import re
import json
import time
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, helpers

# Configuration
MATCH_ID = 'nz_ind_super_over_2020'
INDEX_NAME = f"highlights_{MATCH_ID}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
METADATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'metadata')
INPUT_JSON = os.path.join(METADATA_DIR, f'{MATCH_ID}_tags.json')

# Initialize the embedding model
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenSearch client (Security disabled for local dev)
print("Connecting to OpenSearch...")
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

# The ASR Mapping Dictionary
# Keep keys at least 3 letters long: very short keys (e.g. "ro", "sa") match ordinary
# words and syllables in the commentary and corrupt the transcript.
ASR_CORRECTIONS = {
    "bumra": "Bumrah",
    "boomerra": "Bumrah",
    "coli": "Kohli",
    "coley": "Kohli",
    "kohley": "Kohli",
    "ciphert": "Seifert",
    "ciphford": "Seifert",
    "guptal": "Guptill",
    "gupt": "Guptill",
    "gabul": "Guptill",
    "jadesia": "Jadeja",
    "jadeia": "Jadeja",
    "sally": "Southee", 
    "salvi": "Southee",
    "saudi": "Southee",
    "k rail": "KL Rahul",
    "kra": "KL Rahul",
    "takur": "Thakur"
}

def clean_asr_text(text: str) -> str:
    """Uses regex word boundaries to instantly replace bad YouTube captions with correct names."""
    if not text:
        return text
        
    cleaned_text = text
    for wrong, right in ASR_CORRECTIONS.items():
        # \b ensures we only match whole words, IGNORECASE handles Coli vs coli
        cleaned_text = re.sub(rf'\b{wrong}\b', right, cleaned_text, flags=re.IGNORECASE)
    return cleaned_text

def create_index():
    """Creates a k-NN (k-Nearest Neighbors) index in OpenSearch to store vectors."""
    index_body = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                "chunk_index": {"type": "integer"},
                "filename": {"type": "keyword"},
                "event_type": {"type": "keyword"},
                "is_highlight": {"type": "boolean"},
                "description": {"type": "text"},
                # The crucial field: A dense vector of 384 dimensions to match our model
                "vector_embedding": {
                    "type": "knn_vector",
                    "dimension": 384, 
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }

    if client.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} already exists. Deleting it to start fresh.")
        client.indices.delete(index=INDEX_NAME)

    client.indices.create(index=INDEX_NAME, body=index_body)
    print(f"Created k-NN index: {INDEX_NAME}")

def ingest_data():
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: Cannot find {INPUT_JSON}. Run the vision tagger first.")
        return

    with open(INPUT_JSON, 'r') as f:
        records = json.load(f)

    print(f"Found {len(records)} records. Generating embeddings...")
    actions = []
    
    # Index EVERY chunk. The vision model labels most chunks "normal_play" (even sixes and
    # wickets), so filtering on its labels hid ~85% of the match from search. The commentary
    # carries most of the signal; the Cross-Encoder at query time does the filtering instead.
    texts = []
    for record in records:
        visual_description = record.get('description', '').strip().rstrip('.')
        cleaned_audio = clean_asr_text(record.get('audio_context', ''))
        texts.append(f"{visual_description}. Commentary: {cleaned_audio}")

    # Encoding in one batch is much faster than one call per chunk
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

    for record, text_to_embed, vector in zip(records, texts, vectors):
        actions.append({
            "_index": INDEX_NAME,
            "_id": record['chunk_index'],
            "_source": {
                "chunk_index": record['chunk_index'],
                "filename": record['filename'],
                "event_type": record.get('event_type', 'normal_play'),
                "is_highlight": record.get('is_highlight', False),
                "description": text_to_embed,
                "vector_embedding": vector.tolist()
            }
        })

    if actions:
        print(f"Bulk ingesting {len(actions)} vectors into OpenSearch...")
        helpers.bulk(client, actions)
        # Give the database a second to settle
        time.sleep(1) 
        print("Ingestion complete!")
    else:
        print("No records found in JSON to ingest.")

if __name__ == "__main__":
    create_index()
    ingest_data()