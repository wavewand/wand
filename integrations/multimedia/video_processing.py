"""
Video processing integrations for Wand
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Optional imports for video processing
try:
    import cv2
    import numpy as np

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class FFmpegIntegration(BaseIntegration):
    """FFmpeg video processing integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "ffmpeg_path": "ffmpeg",
            "ffprobe_path": "ffprobe",
            "temp_dir": tempfile.gettempdir(),
            "max_file_size": 1024 * 1024 * 1024,  # 1GB
            "timeout": 300,  # 5 minutes
            "quality": "medium",
        }
        super().__init__("ffmpeg", {**default_config, **(config or {})})

    async def initialize(self):
        """Check if FFmpeg is available"""
        try:
            result = await self._run_command([self.config["ffmpeg_path"], "-version"])
            logger.info("✅ FFmpeg initialized successfully")
        except Exception as e:
            logger.error(f"❌ FFmpeg initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup temporary files"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check FFmpeg health"""
        try:
            result = await self._run_command([self.config["ffmpeg_path"], "-version"])
            return {"status": "healthy", "version": result.stdout.split('\n')[0] if result.stdout else "unknown"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute FFmpeg operations"""

        if operation == "convert":
            return await self._convert_video(**kwargs)
        elif operation == "compress":
            return await self._compress_video(**kwargs)
        elif operation == "extract_audio":
            return await self._extract_audio(**kwargs)
        elif operation == "generate_thumbnail":
            return await self._generate_thumbnail(**kwargs)
        elif operation == "get_info":
            return await self._get_video_info(**kwargs)
        elif operation == "trim":
            return await self._trim_video(**kwargs)
        elif operation == "merge":
            return await self._merge_videos(**kwargs)
        elif operation == "add_watermark":
            return await self._add_watermark(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _convert_video(self, input_path: str, output_path: str, format: str = "mp4", **options) -> Dict[str, Any]:
        """Convert video to different format"""
        cmd = [self.config["ffmpeg_path"], "-i", input_path, "-y"]  # Overwrite output file

        # Add format-specific options
        if format.lower() == "mp4":
            cmd.extend(["-c:v", "libx264", "-c:a", "aac"])
        elif format.lower() == "webm":
            cmd.extend(["-c:v", "libvpx-vp9", "-c:a", "libopus"])
        elif format.lower() == "avi":
            cmd.extend(["-c:v", "libxvid", "-c:a", "mp3"])

        # Add quality settings
        quality = options.get("quality", self.config["quality"])
        if quality == "high":
            cmd.extend(["-crf", "18"])
        elif quality == "medium":
            cmd.extend(["-crf", "23"])
        elif quality == "low":
            cmd.extend(["-crf", "28"])

        cmd.append(output_path)

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "format": format, "command": " ".join(cmd)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _compress_video(
        self, input_path: str, output_path: str, target_size_mb: Optional[int] = None, crf: int = 23
    ) -> Dict[str, Any]:
        """Compress video file"""
        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-y",
            output_path,
        ]

        if target_size_mb:
            # Calculate bitrate for target file size
            duration = await self._get_duration(input_path)
            if duration:
                target_bitrate = int((target_size_mb * 8 * 1024) / duration)
                cmd.extend(["-b:v", f"{target_bitrate}k"])

        try:
            result = await self._run_command(cmd)

            # Get file sizes
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            compression_ratio = (input_size - output_size) / input_size * 100

            return {
                "success": True,
                "output_path": output_path,
                "input_size_mb": input_size / (1024 * 1024),
                "output_size_mb": output_size / (1024 * 1024),
                "compression_ratio": f"{compression_ratio:.1f}%",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_audio(self, input_path: str, output_path: str, format: str = "mp3") -> Dict[str, Any]:
        """Extract audio from video"""
        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            input_path,
            "-vn",  # No video
            "-acodec",
            "libmp3lame" if format == "mp3" else "copy",
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "format": format}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_thumbnail(
        self, input_path: str, output_path: str, timestamp: str = "00:00:01"
    ) -> Dict[str, Any]:
        """Generate thumbnail from video"""
        cmd = [self.config["ffmpeg_path"], "-i", input_path, "-ss", timestamp, "-vframes", "1", "-y", output_path]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "timestamp": timestamp}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_video_info(self, input_path: str) -> Dict[str, Any]:
        """Get video file information"""
        cmd = [
            self.config["ffprobe_path"],
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            input_path,
        ]

        try:
            result = await self._run_command(cmd)
            import json

            info = json.loads(result.stdout)

            # Extract useful information
            video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
            audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)

            return {
                "success": True,
                "duration": float(info["format"].get("duration", 0)),
                "size": int(info["format"].get("size", 0)),
                "bitrate": int(info["format"].get("bit_rate", 0)),
                "video": {
                    "codec": video_stream.get("codec_name") if video_stream else None,
                    "width": int(video_stream.get("width", 0)) if video_stream else None,
                    "height": int(video_stream.get("height", 0)) if video_stream else None,
                    "fps": self._parse_framerate(video_stream.get("r_frame_rate", "0/1")) if video_stream else None,
                },
                "audio": {
                    "codec": audio_stream.get("codec_name") if audio_stream else None,
                    "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
                    "channels": int(audio_stream.get("channels", 0)) if audio_stream else None,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_framerate(self, framerate_str: str) -> Optional[float]:
        """Parse framerate string (e.g., '30/1' or '25') to float"""
        try:
            if '/' in framerate_str:
                numerator, denominator = framerate_str.split('/')
                return float(numerator) / float(denominator) if float(denominator) != 0 else None
            else:
                return float(framerate_str)
        except (ValueError, ZeroDivisionError):
            return None

    async def _trim_video(
        self, input_path: str, output_path: str, start_time: str, duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trim video"""
        cmd = [self.config["ffmpeg_path"], "-i", input_path, "-ss", start_time]

        if duration:
            cmd.extend(["-t", duration])

        cmd.extend(["-c", "copy", "-y", output_path])

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "start_time": start_time, "duration": duration}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _merge_videos(self, input_paths: List[str], output_path: str) -> Dict[str, Any]:
        """Merge multiple videos"""
        # Create temporary file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in input_paths:
                f.write(f"file '{path}'\n")
            filelist_path = f.name

        cmd = [
            self.config["ffmpeg_path"],
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            filelist_path,
            "-c",
            "copy",
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            os.unlink(filelist_path)  # Clean up temp file

            return {"success": True, "output_path": output_path, "input_files": len(input_paths)}
        except Exception as e:
            if os.path.exists(filelist_path):
                os.unlink(filelist_path)
            return {"success": False, "error": str(e)}

    async def _add_watermark(
        self, input_path: str, output_path: str, watermark_path: str, position: str = "bottom-right"
    ) -> Dict[str, Any]:
        """Add watermark to video"""
        # Position mapping
        positions = {
            "top-left": "10:10",
            "top-right": "W-w-10:10",
            "bottom-left": "10:H-h-10",
            "bottom-right": "W-w-10:H-h-10",
            "center": "(W-w)/2:(H-h)/2",
        }

        overlay_position = positions.get(position, positions["bottom-right"])

        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            input_path,
            "-i",
            watermark_path,
            "-filter_complex",
            f"[0:v][1:v]overlay={overlay_position}",
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "watermark": watermark_path, "position": position}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run FFmpeg command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config["timeout"])

            result = subprocess.CompletedProcess(cmd, process.returncode, stdout.decode(), stderr.decode())

            if result.returncode != 0:
                raise Exception(f"FFmpeg command failed: {result.stderr}")

            return result

        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"FFmpeg command timed out after {self.config['timeout']} seconds")

    async def _get_duration(self, input_path: str) -> Optional[float]:
        """Get video duration in seconds"""
        try:
            info = await self._get_video_info(input_path)
            return info.get("duration") if info.get("success") else None
        except BaseException:
            return None


class OpenCVIntegration(BaseIntegration):
    """OpenCV computer vision integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not HAS_OPENCV:
            raise ImportError("OpenCV is not installed. Install with: pip install opencv-python")
        default_config = {
            "cascade_path": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            "confidence_threshold": 0.5,
        }
        super().__init__("opencv", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize OpenCV"""
        logger.info("✅ OpenCV initialized successfully")

    async def cleanup(self):
        """Cleanup OpenCV resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenCV health"""
        return {"status": "healthy", "version": cv2.__version__}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute OpenCV operations"""

        if operation == "detect_faces":
            return await self._detect_faces(**kwargs)
        elif operation == "detect_objects":
            return await self._detect_objects(**kwargs)
        elif operation == "extract_frames":
            return await self._extract_frames(**kwargs)
        elif operation == "motion_detection":
            return await self._motion_detection(**kwargs)
        elif operation == "blur_faces":
            return await self._blur_faces(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _detect_faces(self, image_path: str) -> Dict[str, Any]:
        """Detect faces in image"""
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Could not load image"}

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load the face cascade
            face_cascade = cv2.CascadeClassifier(self.config["cascade_path"])

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            face_data = []
            for x, y, w, h in faces:
                face_data.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "confidence": 1.0,  # Haar cascades don't provide confidence
                    }
                )

            return {
                "success": True,
                "faces_detected": len(faces),
                "faces": face_data,
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_frames(self, video_path: str, output_dir: str, interval: int = 30) -> Dict[str, Any]:
        """Extract frames from video"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Could not open video"}

            os.makedirs(output_dir, exist_ok=True)

            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame every 'interval' frames
                if frame_count % interval == 0:
                    output_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    cv2.imwrite(output_path, frame)
                    extracted_count += 1

                frame_count += 1

            cap.release()

            return {
                "success": True,
                "total_frames": frame_count,
                "extracted_frames": extracted_count,
                "output_directory": output_dir,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _blur_faces(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Blur faces in image or video"""
        try:
            # Check if input is image or video
            if input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                return await self._blur_faces_image(input_path, output_path)
            else:
                return await self._blur_faces_video(input_path, output_path)

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _blur_faces_image(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Blur faces in image"""
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            return {"success": False, "error": "Could not load image"}

        # Detect faces
        face_result = await self._detect_faces(input_path)
        if not face_result["success"]:
            return face_result

        # Blur each face
        for face in face_result["faces"]:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            face_region = image[y : y + h, x : x + w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            image[y : y + h, x : x + w] = blurred_face

        # Save result
        cv2.imwrite(output_path, image)

        return {"success": True, "faces_blurred": len(face_result["faces"]), "output_path": output_path}


class YouTubeIntegration(BaseIntegration):
    """YouTube API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("YOUTUBE_API_KEY", ""),
            "base_url": "https://www.googleapis.com/youtube/v3",
        }
        super().__init__("youtube", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize YouTube API"""
        if not self.config["api_key"]:
            logger.warning("⚠️  YouTube API key not configured")
        else:
            logger.info("✅ YouTube integration initialized")

    async def cleanup(self):
        """Cleanup YouTube resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check YouTube API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            # Test API with a simple search
            async with aiohttp.ClientSession() as session:
                params = {"part": "snippet", "q": "test", "maxResults": 1, "key": self.config["api_key"]}
                async with session.get(f"{self.config['base_url']}/search", params=params) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute YouTube operations"""

        if operation == "search":
            return await self._search_videos(**kwargs)
        elif operation == "get_video_info":
            return await self._get_video_info(**kwargs)
        elif operation == "download":
            return await self._download_video(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _search_videos(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search YouTube videos"""
        if not self.config["api_key"]:
            return {"success": False, "error": "YouTube API key not configured"}

        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "part": "snippet",
                    "q": query,
                    "maxResults": max_results,
                    "type": "video",
                    "key": self.config["api_key"],
                }

                async with session.get(f"{self.config['base_url']}/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        videos = []
                        for item in data.get("items", []):
                            videos.append(
                                {
                                    "video_id": item["id"]["videoId"],
                                    "title": item["snippet"]["title"],
                                    "description": item["snippet"]["description"],
                                    "channel": item["snippet"]["channelTitle"],
                                    "published_at": item["snippet"]["publishedAt"],
                                    "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
                                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                                }
                            )

                        return {
                            "success": True,
                            "query": query,
                            "total_results": data.get("pageInfo", {}).get("totalResults", 0),
                            "videos": videos,
                        }
                    else:
                        error_data = await response.json()
                        return {"success": False, "error": error_data.get("error", {}).get("message", "Unknown error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class TwitchIntegration(BaseIntegration):
    """Twitch API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "client_id": os.getenv("TWITCH_CLIENT_ID", ""),
            "client_secret": os.getenv("TWITCH_CLIENT_SECRET", ""),
            "base_url": "https://api.twitch.tv/helix",
        }
        super().__init__("twitch", {**default_config, **(config or {})})
        self.access_token = None

    async def initialize(self):
        """Initialize Twitch API"""
        if not self.config["client_id"] or not self.config["client_secret"]:
            logger.warning("⚠️  Twitch credentials not configured")
        else:
            await self._get_access_token()
            logger.info("✅ Twitch integration initialized")

    async def cleanup(self):
        """Cleanup Twitch resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Twitch API health"""
        if not self.access_token:
            return {"status": "unhealthy", "error": "Not authenticated"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Client-ID": self.config["client_id"], "Authorization": f"Bearer {self.access_token}"}
                async with session.get(f"{self.config['base_url']}/users", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_access_token(self):
        """Get OAuth access token"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "client_id": self.config["client_id"],
                    "client_secret": self.config["client_secret"],
                    "grant_type": "client_credentials",
                }
                async with session.post("https://id.twitch.tv/oauth2/token", data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data["access_token"]
        except Exception as e:
            logger.error(f"Failed to get Twitch access token: {e}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Twitch operations"""

        if operation == "get_streams":
            return await self._get_streams(**kwargs)
        elif operation == "get_clips":
            return await self._get_clips(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}


class ZoomIntegration(BaseIntegration):
    """Zoom API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("ZOOM_API_KEY", ""),
            "api_secret": os.getenv("ZOOM_API_SECRET", ""),
            "base_url": "https://api.zoom.us/v2",
        }
        super().__init__("zoom", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Zoom API"""
        logger.info("✅ Zoom integration initialized")

    async def cleanup(self):
        """Cleanup Zoom resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Zoom API health"""
        return {"status": "healthy"}  # Simplified for demo

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Zoom operations"""

        if operation == "list_recordings":
            return await self._list_recordings(**kwargs)
        elif operation == "download_recording":
            return await self._download_recording(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _list_recordings(self, user_id: str = "me") -> Dict[str, Any]:
        """List user recordings"""
        # Simplified implementation
        return {
            "success": True,
            "recordings": [
                {
                    "id": "123456789",
                    "topic": "Sample Meeting",
                    "start_time": "2024-01-01T10:00:00Z",
                    "duration": 3600,
                    "file_size": 1024000,
                }
            ],
        }
