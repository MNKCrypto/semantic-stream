import os
import json
import re
import subprocess
import ollama
from pydantic import BaseModel

# Configuration
MATCH_ID = 'nz_ind_super_over_2020'
MODEL_NAME = 'llava'
CHUNK_DURATION = 6  # Updated to 6 seconds per chunk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CDN_MOCK_DIR = os.path.join(PROJECT_ROOT, 'data', 'cdn_mock', MATCH_ID)
METADATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'metadata')
OUTPUT_JSON = os.path.join(METADATA_DIR, f'{MATCH_ID}_tags.json')
TEMP_FRAME = os.path.join(METADATA_DIR, 'temp_frame.jpg')
SUBTITLE_PATH = os.path.join(METADATA_DIR, f'{MATCH_ID}.vtt')

class VideoHighlight(BaseModel):
    is_highlight: bool
    event_type: str
    description: str

def parse_vtt_time(time_str: str) -> float:
    parts = time_str.replace(',', '.').split(':')
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    return (hours * 3600) + (minutes * 60) + seconds

def get_subtitle_context(chunk_index: int) -> str:
    """Extracts subtitles that fall precisely within this 6-second chunk."""
    if not os.path.exists(SUBTITLE_PATH):
        return ""
        
    start_time = chunk_index * CHUNK_DURATION
    end_time = start_time + CHUNK_DURATION
    
    with open(SUBTITLE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Strip YouTube inline word timestamps
    content = re.sub(r'<[^>]+>', '', content)
    lines = content.split('\n')
    
    context_lines = []
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            try:
                start_str, end_str = lines[i].split(' --> ')
                b_start = parse_vtt_time(start_str.strip())
                b_end = parse_vtt_time(end_str.strip().split(' ')[0])
                b_mid = (b_start + b_end) / 2.0
                
                text_buffer = []
                i += 1
                while i < len(lines) and '-->' not in lines[i]:
                    clean = lines[i].replace('&gt;&gt;', '').replace('>>', '').replace('&nbsp;', ' ').strip()
                    if clean and clean not in ["WEBVTT", "Kind: captions", "Language: en"]:
                        text_buffer.append(clean)
                    i += 1
                
                # Check if the subtitle's midpoint falls strictly inside our 6-second window
                if start_time <= b_mid < end_time:
                    context_lines.extend(text_buffer)
                continue
            except Exception:
                pass
        i += 1
        
    # Deduplicate rolling lines
    final_text = []
    for line in context_lines:
        if not final_text or (line not in final_text[-1] and final_text[-1] not in line):
            final_text.append(line)
            
    return " ".join(final_text)

def extract_middle_frame(chunk_path: str, output_path: str):
    command = [
        'ffmpeg', '-y',
        '-i', chunk_path,
        '-ss', '00:00:03', # Grab the frame right in the middle of the 6-second chunk
        '-vframes', '1',
        '-q:v', '2',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_chunk(chunk_path: str, chunk_index: int) -> dict:
    print(f"\n--- Processing chunk {chunk_index} ---")
    
    extract_middle_frame(chunk_path, TEMP_FRAME)
    if not os.path.exists(TEMP_FRAME):
        print("ERROR: Failed to extract frame.")
        return None

    # Get the audio context for this exact 6-second window
    audio_context = get_subtitle_context(chunk_index)

    print(f"Audio Context: '{audio_context}'")

    # True Decoupling: The model stays deaf to avoid NLP hallucinations
    prompt = """
    Describe the primary physical action happening in this cricket video frame.
    Write exactly ONE short, simple sentence (under 10 words).
    Focus ONLY on the physical movement (e.g., 'Batsman hitting ball', 'Bowler delivering ball', 'Fielder catching ball', 'Umpire raising finger', 'Players celebrating').
    Do NOT use player names. Do NOT use generic phrases like "The image shows".
    
    Valid event_types: ['wicket', 'boundary', 'catch', 'celebration', 'crowd_reaction', 'normal_play']
    Respond ONLY with raw JSON matching the schema.
    """

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [TEMP_FRAME]
            }],
            format=VideoHighlight.model_json_schema(),
            options={'temperature': 0.1}
        )
        
        cleaned = response['message']['content'].replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
        
        # Python handles the fusion, safely passing the 6-second audio string to the JSON
        return {
            "chunk_index": chunk_index,
            "filename": os.path.basename(chunk_path),
            "is_highlight": result.get('is_highlight', False),
            "event_type": result.get('event_type', 'normal_play'),
            "description": result.get('description', ''),
            "audio_context": audio_context
        }
    except Exception as e:
        print(f"ERROR: Ollama inference failed: {e}")
        return None

def main():
    os.makedirs(METADATA_DIR, exist_ok=True)
    all_chunks = sorted([f for f in os.listdir(CDN_MOCK_DIR) if f.endswith('.ts')])
    
    database_records = []
    
    for i, chunk_filename in enumerate(all_chunks):
        chunk_path = os.path.join(CDN_MOCK_DIR, chunk_filename)
        data = analyze_chunk(chunk_path, i)
        if data:
            database_records.append(data)
            print(f"Saved: {data['description']}")
            
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(database_records, f, indent=4)
        
    if os.path.exists(TEMP_FRAME):
        os.remove(TEMP_FRAME)
        
    print(f"\nFinished! Metadata saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()