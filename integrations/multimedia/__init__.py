"""
🎬 Multimedia Processing Integrations

Video, audio, and image processing capabilities for Wand
"""

from .audio_processing import (
    AudioIntegration,
    ElevenLabsIntegration,
    PodcastIntegration,
    SpotifyIntegration,
    WhisperIntegration,
)
from .image_processing import ChartIntegration, ImageIntegration, OCRIntegration, QRIntegration
from .video_processing import (
    FFmpegIntegration,
    OpenCVIntegration,
    TwitchIntegration,
    YouTubeIntegration,
    ZoomIntegration,
)

# Initialize integration instances
ffmpeg_integration = FFmpegIntegration()

# OpenCV requires optional dependencies
try:
    opencv_integration = OpenCVIntegration()
except ImportError:
    opencv_integration = None

youtube_integration = YouTubeIntegration()
twitch_integration = TwitchIntegration()
zoom_integration = ZoomIntegration()

# Audio integrations - some require optional dependencies
try:
    audio_integration = AudioIntegration()
except ImportError:
    audio_integration = None

try:
    whisper_integration = WhisperIntegration()
except ImportError:
    whisper_integration = None

elevenlabs_integration = ElevenLabsIntegration()
spotify_integration = SpotifyIntegration()
podcast_integration = PodcastIntegration()

# Image integrations - some require optional dependencies
try:
    image_integration = ImageIntegration()
except ImportError:
    image_integration = None

try:
    ocr_integration = OCRIntegration()
except ImportError:
    ocr_integration = None

qr_integration = QRIntegration()

try:
    chart_integration = ChartIntegration()
except ImportError:
    chart_integration = None

__all__ = [
    # Video processing
    "ffmpeg_integration",
    "opencv_integration",
    "youtube_integration",
    "twitch_integration",
    "zoom_integration",
    # Audio processing
    "audio_integration",
    "whisper_integration",
    "elevenlabs_integration",
    "spotify_integration",
    "podcast_integration",
    # Image processing
    "image_integration",
    "ocr_integration",
    "qr_integration",
    "chart_integration",
]
