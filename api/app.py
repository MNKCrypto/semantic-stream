import os
import ollama
from litestar import Litestar, get, Response
from litestar.config.cors import CORSConfig
from litestar.static_files.config import StaticFilesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from opensearchpy import OpenSearch

# Configuration
MATCH_ID = 'nz_ind_super_over_2020'
INDEX_NAME = f"highlights_{MATCH_ID}"
CDN_BASE_URL = "http://localhost:8000/cdn"

CHUNK_DURATION = 6          # Seconds per .ts segment (must match indexer/download_chunk.py)
CANDIDATE_POOL = 30         # Chunks fetched from OpenSearch before re-ranking
TOP_HITS = 5                # Best re-ranked chunks kept as "core" moments
SCORE_THRESHOLD = -3.0      # Cross-Encoder logit floor
PAD_BEFORE = 1              # Chunks of context before each hit (run-up)
PAD_AFTER = 1               # Chunks of context after each hit (reaction)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CDN_MOCK_DIR = os.path.join(PROJECT_ROOT, 'data', 'cdn_mock', MATCH_ID)

# Highest chunk index that actually exists on the mock CDN, so padding never points past the end
MAX_CHUNK_INDEX = len([f for f in os.listdir(CDN_MOCK_DIR) if f.endswith('.ts')]) - 1

print("Loading Bi-Encoder for Coarse Retrieval...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading Cross-Encoder for Post-Search Re-ranking...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Connecting API to OpenSearch...")
client = OpenSearch([{'host': 'localhost', 'port': 9200}], use_ssl=False, verify_certs=False)


def generate_m3u8(chunk_indices: list[int]) -> str:
    manifest = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{CHUNK_DURATION}",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]

    last_idx = None
    for idx in chunk_indices:
        # Only non-adjacent jumps need a discontinuity (not the very first segment)
        if last_idx is not None and idx != last_idx + 1:
            manifest.append("#EXT-X-DISCONTINUITY")
        manifest.append(f"#EXTINF:{CHUNK_DURATION:.1f},")
        manifest.append(f"{CDN_BASE_URL}/{MATCH_ID}_{idx:03d}.ts")
        last_idx = idx

    manifest.append("#EXT-X-ENDLIST")
    return "\n".join(manifest)


def pad_indices(core_indices: list[int]) -> list[int]:
    """Expands each hit into a short continuous play, clamped to chunks that exist."""
    expanded = set()
    for idx in core_indices:
        for offset in range(-PAD_BEFORE, PAD_AFTER + 1):
            expanded.add(min(MAX_CHUNK_INDEX, max(0, idx + offset)))
    return sorted(expanded)


def rewrite_query(raw_query: str) -> str:
    prompt = f"""
    A user searched for: "{raw_query}"
    Rewrite this into a clean search query.

    CRITICAL RULES:
    1. If the user ONLY provides a player name (e.g., "KL Rahul"), DO NOT invent an action like "drives" or "hits". Just return the name.
    2. Do NOT use flowery language or commentary style.
    3. Respond ONLY with the rewritten query.
    """
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        expanded_query = response['message']['content'].strip(' "\'')
        print(f"Pre-Search Rewrite: '{raw_query}' -> '{expanded_query}'")
        return expanded_query
    except Exception as e:
        print(f"Query rewrite failed: {e}")
        return raw_query


# The handler is synchronous because Ollama, the encoders and the OpenSearch client all block.
# sync_to_thread=True runs it in a worker thread so one slow search doesn't freeze the event loop
# (and the /cdn static file serving) for every other user.
@get("/search", sync_to_thread=True)
def search_highlights(q: str) -> Response[str]:
    print(f"\n--- New Search Request: '{q}' ---")

    # 1. PRE-SEARCH (Query Optimization)
    optimized_query = rewrite_query(q)

    # 2. EMBEDDING
    query_vector = model.encode(optimized_query).tolist()

    # 3. COARSE RETRIEVAL (OpenSearch k-NN)
    # Cast a wide net because the Cross-Encoder will aggressively filter.
    search_body = {
        "size": CANDIDATE_POOL,
        "query": {
            "knn": {
                "vector_embedding": {
                    "vector": query_vector,
                    "k": CANDIDATE_POOL
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=search_body)
    hits = response['hits']['hits']

    if not hits:
        return Response(content="No highlights found", status_code=404, media_type="text/plain")

    # 4. POST-SEARCH (Cross-Encoder Re-ranking)
    print(f"Scoring {len(hits)} candidates with Cross-Encoder...")
    pairs = [[optimized_query, hit['_source']['description']] for hit in hits]
    scores = cross_encoder.predict(pairs)
    for hit, score in zip(hits, scores):
        hit['cross_encoder_score'] = float(score)

    # 5. FILTER & SORT
    strong_hits = [hit for hit in hits if hit['cross_encoder_score'] > SCORE_THRESHOLD]

    # If the Cross-Encoder rejected everything, return a 404 instead of an empty playlist
    if not strong_hits:
        print("Cross-Encoder found no relevant chunks.")
        return Response(content="No highlights found", status_code=404, media_type="text/plain")

    best_hits = sorted(strong_hits, key=lambda x: x['cross_encoder_score'], reverse=True)[:TOP_HITS]
    core_indices = [hit['_source']['chunk_index'] for hit in best_hits]

    # 6. TEMPORAL PADDING
    chunk_indices = pad_indices(core_indices)
    print(f"Padded into a {len(chunk_indices) * CHUNK_DURATION}-second highlight reel.")

    m3u8_content = generate_m3u8(chunk_indices)
    return Response(content=m3u8_content, media_type="application/vnd.apple.mpegurl")


static_files = StaticFilesConfig(directories=[CDN_MOCK_DIR], path="/cdn")
cors_config = CORSConfig(allow_origins=["*"])

app = Litestar(
    route_handlers=[search_highlights],
    static_files_config=[static_files],
    cors_config=cors_config
)
