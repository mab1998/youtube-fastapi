from pytube import YouTube, exceptions as pytube_exceptions
import os
from fastapi import HTTPException
from moviepy.editor import AudioFileClip

class YouTubeDownloader:
    def __init__(self, download_path: str = "./downloads"):
        self.download_path = download_path
        
    def get_video_info(self, url: str) -> dict:
        try:
            yt = YouTube(url)
            video_info = {
                'title': yt.title,
                'description': yt.description
            }
            return video_info
        except pytube_exceptions.PytubeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def download_youtube_audio(self, url: str) -> str:
        try:
            yt = YouTube(url)
            video = yt.streams.filter(only_audio=True).first()
            out_file = video.download(output_path=self.download_path)
            base, ext = os.path.splitext(out_file)
            new_file = base + '.mp3'

            # Convert to MP3 and cleanup
            audio_clip = AudioFileClip(out_file)
            audio_clip.write_audiofile(new_file)
            audio_clip.close()
            # os.remove(out_file)  # Remove original download
            
            return new_file            
            
            
            
            
            
        except pytube_exceptions.PytubeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    def check_video_length( url: str) -> bool:
        yt = YouTube(url)
        video_duration = yt.length  # Duration in seconds
        return video_duration <= 600  # True if 10 minutes or less
