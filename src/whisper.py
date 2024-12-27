import openai
import os
from pathlib import Path

os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_BASE_URL'] = ''

from pathlib import Path
import openai
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

thread_local = threading.local()

def get_openai_client():
    """
    为每个线程获取一个OpenAI客户端实例
    """
    if not hasattr(thread_local, "client"):
        thread_local.client = openai.OpenAI()
    return thread_local.client

def transcribe_and_save(audio_path, output_dir="text"):
    """
    Transcribe an audio file and save the result to a text file
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save the transcript (default: 'text')
    """
    try:
        client = get_openai_client()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_name = Path(audio_path).stem
        output_path = Path(output_dir) / f"{base_name}_transcript.txt"
        
        # Check if file already exists
        if output_path.exists():
            return True, f"Skipped {audio_path} (already processed)"
        
        # Open and transcribe audio file
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Save transcription to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
            
        return True, f"Successfully processed {audio_path}"
        
    except Exception as e:
        return False

def process_all_wav_files(input_dir, output_dir="text", max_workers=4):
    """
    Process all WAV files in the specified directory using multiple threads
    
    Args:
        input_dir (str): Directory containing WAV files
        output_dir (str): Directory to save the transcripts (default: 'text')
        max_workers (int): Maximum number of worker threads (default: 4)
    """
    # Convert input directory to Path object
    input_path = Path(input_dir)
    
    # Get list of all .wav files in the directory
    wav_files = list(input_path.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(transcribe_and_save, str(wav_file), output_dir): wav_file
            for wav_file in wav_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(wav_files), desc="Processing WAV files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                wav_file = future_to_file[future]
                try:
                    success, message = future.result()
                    if not success:
                        tqdm.write(message)
                except Exception as e:
                    pass
                finally:
                    pbar.update(1)
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)

if __name__ == "__main__":
    # Example usage
    input_directory = "./Deception/CBU0521DD_stories"
    output_directory = "./text"
    
    # 设置并发线程数
    num_threads = 10  # 可以根据需要调整
    
    process_all_wav_files(
        input_directory, 
        output_directory, 
        max_workers=num_threads
    )
