import os
import yt_dlp
import ffmpeg

# Configuration
YOUTUBE_URL = 'https://www.youtube.com/watch?v=oyBy9YXLlkQ'
MATCH_ID = 'nz_ind_super_over_2020'
CHUNK_DURATION = 6  # Seconds per segment (api/app.py and vision_tagger.py assume the same value)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_videos')
CDN_MOCK_DIR = os.path.join(PROJECT_ROOT, 'data', 'cdn_mock', MATCH_ID)
METADATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'metadata')

RAW_VIDEO_PATH = os.path.join(RAW_VIDEO_DIR, f'{MATCH_ID}.mp4')
SUBTITLE_PATH_TEMPLATE = os.path.join(METADATA_DIR, f'{MATCH_ID}')

def download_video_and_subs():
    print(f"Starting download to {RAW_VIDEO_PATH}...")
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': RAW_VIDEO_PATH,
        'merge_output_format': 'mp4',
        # Subtitle extraction configuration
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([YOUTUBE_URL])
        
    # yt-dlp saves the sub as {RAW_VIDEO_PATH}.en.vtt in the video folder by default.
    # Let's move it to our metadata folder so it's clean.
    downloaded_sub = os.path.join(RAW_VIDEO_DIR, f'{MATCH_ID}.en.vtt')
    target_sub = os.path.join(METADATA_DIR, f'{MATCH_ID}.vtt')
    if os.path.exists(downloaded_sub):
        os.rename(downloaded_sub, target_sub)
        print(f"Subtitles extracted to {target_sub}")

def chunk_video():
    print(f"Starting HLS chunking into {CDN_MOCK_DIR}...")
    os.makedirs(CDN_MOCK_DIR, exist_ok=True)
    
    # We don't need the master playlist it generates, just the .ts files
    dummy_m3u8 = os.path.join(CDN_MOCK_DIR, 'dummy.m3u8')
    
    try:
        (
            ffmpeg
            .input(RAW_VIDEO_PATH)
            .output(
                dummy_m3u8,
                format='hls',
                hls_time=CHUNK_DURATION,    # 6-second chunks
                hls_list_size=0,            # Keep all chunks (don't create a rolling live window)
                hls_segment_filename=os.path.join(CDN_MOCK_DIR, f'{MATCH_ID}_%03d.ts'),
                vcodec='libx264',
                acodec='aac',
                # Force a keyframe exactly on every segment boundary so each chunk plays on its own
                force_key_frames=f'expr:gte(t,n_forced*{CHUNK_DURATION})'
            )
            .run(overwrite_output=True)
        )
        # Clean up the dummy playlist since Litestar builds it dynamically
        if os.path.exists(dummy_m3u8):
            os.remove(dummy_m3u8)
        print("Successfully chunked video with perfect A/V sync!")
    except ffmpeg.Error as e:
        print(f"FFMPEG Error: {e.stderr.decode()}")

if __name__ == "__main__":
    download_video_and_subs()
    chunk_video()