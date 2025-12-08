from pytube import YouTube
import os

def download_youtube_video(url, output_path='./downloads'):
    """
    Downloads a YouTube video to a specified path using pytube.
    """
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Get the highest resolution stream that has both video and audio (progressive)
        # These are generally up to 720p.
        video_stream = yt.streams.get_highest_resolution()

        if video_stream:
            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)

            print(f"Downloading: {yt.title} to {output_path}...")
            video_stream.download(output_path)
            print("Download completed successfully!")
        else:
            print("Could not find a suitable progressive stream (e.g., 720p or lower with audio).")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Usage Example ---
video_url = input("Enter the YouTube video URL: ")
download_youtube_video(video_url)
# ##################################### ##################################
import yt_dlp
import os
from moviepy.editor import VideoFileClip

def download_and_reencode_for_windows(url, output_dir='./windows_compatible_videos'):
    """
    Downloads the best quality video using yt-dlp, and then re-encodes it 
    with MoviePy to ensure compatibility with Windows Media Player/Photos App.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # yt-dlp options to download the best streams (might download separate audio/video files)
    # Note: yt-dlp might complain about missing system FFmpeg during this download phase 
    # if it needs to merge internally, but we will force a re-encode later anyway.
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best', 
        'outtmpl': os.path.join(output_dir, 'temp_download.%(ext)s'),
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'keepvideo': True, # Keep the files yt-dlp generates
        'progress_hooks': [lambda d: print(f"YT-DLP Status: {d['status']}")],
        'ignoreerrors': 'only_download',
        'no_warnings': True,
    }

    temp_merged_file = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading best quality streams from: {url}...")
            info_dict = ydl.extract_info(url, download=True)
            # Find the path where yt-dlp saved the (potentially merged) temp file
            temp_merged_file = ydl.prepare_filename(info_dict)

        print(f"Download complete: {temp_merged_file}")

        # --- Re-encode with MoviePy for Windows compatibility ---
        print("\nRe-encoding for Windows Media Player compatibility...")
        # MoviePy uses its *own* internal FFmpeg binary which it downloads automatically
        clip = VideoFileClip(temp_merged_file)
        
        final_output_path = os.path.join(output_dir, f"{info_dict.get('title', 'final_video')}.mp4")
        
        # This step forces the H.264 (libx264) and AAC codecs that Windows loves
        clip.write_videofile(
            final_output_path, 
            codec='libx264', 
            audio_codec='aac', 
            temp_audiofile=os.path.join(output_dir, 'temp-audio.m4a'),
            remove_temp=True,
            threads=4,
            preset='medium'
        )
        
        clip.close()
        print(f"\nSuccess! File saved to: {final_output_path}")

    except Exception as e:
        print(f"An error occurred during download or re-encoding: {e}")
        print("Ensure MoviePy had time to download its internal FFmpeg binaries on the first run.")
    finally:
        # Clean up the intermediate yt-dlp file if necessary
        if temp_merged_file and os.path.exists(temp_merged_file):
            # The write_videofile often creates a new file, so the old one can be removed
            try:
                os.remove(temp_merged_file)
                print(f"Cleaned up temporary download file.")
            except OSError as e:
                print(f"Error removing temp file: {e}")


# --- Usage Example ---
video_url = input("Enter the YouTube video URL: ")
# Example URL for testing: www.youtube.com
download_and_reencode_for_windows(video_url)
