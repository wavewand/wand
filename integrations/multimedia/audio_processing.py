"""
Audio processing integrations for Wand
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

# Optional import for Whisper
try:
    import whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    whisper = None

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class AudioIntegration(BaseIntegration):
    """General audio processing integration using FFmpeg"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "ffmpeg_path": "ffmpeg",
            "temp_dir": tempfile.gettempdir(),
            "timeout": 300,
            "supported_formats": ["mp3", "wav", "flac", "aac", "ogg", "m4a"],
        }
        super().__init__("audio", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize audio processing"""
        try:
            result = await self._run_command([self.config["ffmpeg_path"], "-version"])
            logger.info("✅ Audio processing initialized successfully")
        except Exception as e:
            logger.error(f"❌ Audio processing initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup audio resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check audio processing health"""
        try:
            result = await self._run_command([self.config["ffmpeg_path"], "-version"])
            return {"status": "healthy", "ffmpeg_available": True}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute audio operations"""

        if operation == "convert":
            return await self._convert_audio(**kwargs)
        elif operation == "extract_from_video":
            return await self._extract_from_video(**kwargs)
        elif operation == "merge":
            return await self._merge_audio(**kwargs)
        elif operation == "trim":
            return await self._trim_audio(**kwargs)
        elif operation == "normalize":
            return await self._normalize_audio(**kwargs)
        elif operation == "add_effects":
            return await self._add_effects(**kwargs)
        elif operation == "get_info":
            return await self._get_audio_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _convert_audio(
        self, input_path: str, output_path: str, format: str = "mp3", bitrate: str = "192k"
    ) -> Dict[str, Any]:
        """Convert audio format"""
        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            input_path,
            "-codec:a",
            self._get_codec_for_format(format),
            "-b:a",
            bitrate,
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "format": format, "bitrate": bitrate}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_from_video(self, video_path: str, output_path: str, format: str = "mp3") -> Dict[str, Any]:
        """Extract audio from video file"""
        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            video_path,
            "-vn",  # No video
            "-codec:a",
            self._get_codec_for_format(format),
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "source_video": video_path, "format": format}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _merge_audio(self, input_paths: List[str], output_path: str) -> Dict[str, Any]:
        """Merge multiple audio files"""
        # Create filter complex for concatenation
        inputs = []
        filter_parts = []

        for i, path in enumerate(input_paths):
            inputs.extend(["-i", path])
            filter_parts.append(f"[{i}:0]")

        filter_complex = f"{''.join(filter_parts)}concat=n={len(input_paths)}:v=0:a=1[out]"

        cmd = (
            [self.config["ffmpeg_path"]]
            + inputs
            + ["-filter_complex", filter_complex, "-map", "[out]", "-y", output_path]
        )

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "input_files": len(input_paths)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _trim_audio(
        self, input_path: str, output_path: str, start_time: str, duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trim audio file"""
        cmd = [self.config["ffmpeg_path"], "-i", input_path, "-ss", start_time]

        if duration:
            cmd.extend(["-t", duration])

        cmd.extend(["-c", "copy", "-y", output_path])

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "start_time": start_time, "duration": duration}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _normalize_audio(self, input_path: str, output_path: str, target_level: str = "-16dB") -> Dict[str, Any]:
        """Normalize audio levels"""
        cmd = [
            self.config["ffmpeg_path"],
            "-i",
            input_path,
            "-filter:a",
            f"loudnorm=I={target_level}",
            "-y",
            output_path,
        ]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "target_level": target_level}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _add_effects(self, input_path: str, output_path: str, effects: List[str]) -> Dict[str, Any]:
        """Add audio effects"""
        # Build filter chain
        filter_chain = ",".join(effects)

        cmd = [self.config["ffmpeg_path"], "-i", input_path, "-filter:a", filter_chain, "-y", output_path]

        try:
            result = await self._run_command(cmd)
            return {"success": True, "output_path": output_path, "effects": effects}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_audio_info(self, input_path: str) -> Dict[str, Any]:
        """Get audio file information"""
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input_path]

        try:
            result = await self._run_command(cmd)
            import json

            info = json.loads(result.stdout)

            audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)

            return {
                "success": True,
                "duration": float(info["format"].get("duration", 0)),
                "size": int(info["format"].get("size", 0)),
                "bitrate": int(info["format"].get("bit_rate", 0)),
                "codec": audio_stream.get("codec_name") if audio_stream else None,
                "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
                "channels": int(audio_stream.get("channels", 0)) if audio_stream else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_codec_for_format(self, format: str) -> str:
        """Get appropriate codec for audio format"""
        codec_map = {
            "mp3": "libmp3lame",
            "aac": "aac",
            "flac": "flac",
            "wav": "pcm_s16le",
            "ogg": "libvorbis",
            "m4a": "aac",
        }
        return codec_map.get(format.lower(), "libmp3lame")

    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run audio processing command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config["timeout"])

            result = subprocess.CompletedProcess(cmd, process.returncode, stdout.decode(), stderr.decode())

            if result.returncode != 0:
                raise Exception(f"Audio command failed: {result.stderr}")

            return result

        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Audio command timed out after {self.config['timeout']} seconds")


class WhisperIntegration(BaseIntegration):
    """OpenAI Whisper speech-to-text integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not HAS_WHISPER:
            raise ImportError("Whisper is not installed. Install with: pip install openai-whisper")
        default_config = {
            "model": "base",  # tiny, base, small, medium, large
            "device": "cpu",  # cpu or cuda
            "language": None,  # Auto-detect if None
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "use_api": False,  # Use OpenAI API instead of local model
        }
        super().__init__("whisper", {**default_config, **(config or {})})
        self.model = None

    async def initialize(self):
        """Initialize Whisper model"""
        try:
            if not self.config["use_api"]:
                # Load local Whisper model
                self.model = whisper.load_model(self.config["model"])
                logger.info(f"✅ Whisper model '{self.config['model']}' loaded successfully")
            else:
                if not self.config["api_key"]:
                    raise Exception("OpenAI API key required for API mode")
                logger.info("✅ Whisper API mode initialized")
        except Exception as e:
            logger.error(f"❌ Whisper initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup Whisper resources"""
        self.model = None

    async def health_check(self) -> Dict[str, Any]:
        """Check Whisper health"""
        if self.config["use_api"]:
            return {
                "status": "healthy" if self.config["api_key"] else "unhealthy",
                "mode": "api",
                "api_configured": bool(self.config["api_key"]),
            }
        else:
            return {"status": "healthy" if self.model else "unhealthy", "mode": "local", "model": self.config["model"]}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Whisper operations"""

        if operation == "transcribe":
            return await self._transcribe_audio(**kwargs)
        elif operation == "translate":
            return await self._translate_audio(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio to text"""
        try:
            if self.config["use_api"]:
                return await self._transcribe_with_api(audio_path, language)
            else:
                return await self._transcribe_with_local_model(audio_path, language)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _transcribe_with_local_model(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe using local Whisper model"""
        if not self.model:
            return {"success": False, "error": "Whisper model not loaded"}

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.model.transcribe(audio_path, language=language or self.config["language"])
        )

        return {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "segments": [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result["segments"]],
        }

    async def _transcribe_with_api(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe using OpenAI API"""
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.config['api_key']}"}

            data = aiohttp.FormData()
            data.add_field('file', open(audio_path, 'rb'))
            data.add_field('model', 'whisper-1')
            if language:
                data.add_field('language', language)
            data.add_field('response_format', 'verbose_json')

            async with session.post(
                "https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "text": result["text"],
                        "language": result.get("language"),
                        "segments": result.get("segments", []),
                    }
                else:
                    error = await response.json()
                    return {"success": False, "error": error.get("error", {}).get("message", "API error")}

    async def _translate_audio(self, audio_path: str) -> Dict[str, Any]:
        """Translate audio to English"""
        try:
            if self.config["use_api"]:
                return await self._translate_with_api(audio_path)
            else:
                return await self._translate_with_local_model(audio_path)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _translate_with_local_model(self, audio_path: str) -> Dict[str, Any]:
        """Translate using local Whisper model"""
        if not self.model:
            return {"success": False, "error": "Whisper model not loaded"}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.model.transcribe(audio_path, task="translate"))

        return {"success": True, "text": result["text"], "source_language": result["language"], "target_language": "en"}


class ElevenLabsIntegration(BaseIntegration):
    """ElevenLabs text-to-speech integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("ELEVENLABS_API_KEY", ""),
            "base_url": "https://api.elevenlabs.io/v1",
            "default_voice_id": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "model_id": "eleven_monolingual_v1",
        }
        super().__init__("elevenlabs", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize ElevenLabs API"""
        if not self.config["api_key"]:
            logger.warning("⚠️  ElevenLabs API key not configured")
        else:
            logger.info("✅ ElevenLabs integration initialized")

    async def cleanup(self):
        """Cleanup ElevenLabs resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check ElevenLabs API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"xi-api-key": self.config["api_key"]}
                async with session.get(f"{self.config['base_url']}/voices", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute ElevenLabs operations"""

        if operation == "text_to_speech":
            return await self._text_to_speech(**kwargs)
        elif operation == "list_voices":
            return await self._list_voices(**kwargs)
        elif operation == "clone_voice":
            return await self._clone_voice(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _text_to_speech(
        self, text: str, voice_id: Optional[str] = None, output_path: str = None
    ) -> Dict[str, Any]:
        """Convert text to speech"""
        if not self.config["api_key"]:
            return {"success": False, "error": "ElevenLabs API key not configured"}

        voice_id = voice_id or self.config["default_voice_id"]

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{int(datetime.now().timestamp())}.mp3")

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"xi-api-key": self.config["api_key"], "Content-Type": "application/json"}

                data = {
                    "text": text,
                    "model_id": self.config["model_id"],
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
                }

                async with session.post(
                    f"{self.config['base_url']}/text-to-speech/{voice_id}", headers=headers, json=data
                ) as response:
                    if response.status == 200:
                        audio_data = await response.read()

                        with open(output_path, 'wb') as f:
                            f.write(audio_data)

                        return {
                            "success": True,
                            "output_path": output_path,
                            "voice_id": voice_id,
                            "text": text,
                            "file_size": len(audio_data),
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("detail", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_voices(self) -> Dict[str, Any]:
        """List available voices"""
        if not self.config["api_key"]:
            return {"success": False, "error": "ElevenLabs API key not configured"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"xi-api-key": self.config["api_key"]}

                async with session.get(f"{self.config['base_url']}/voices", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        voices = []
                        for voice in data.get("voices", []):
                            voices.append(
                                {
                                    "voice_id": voice["voice_id"],
                                    "name": voice["name"],
                                    "description": voice.get("description", ""),
                                    "category": voice.get("category", ""),
                                    "labels": voice.get("labels", {}),
                                }
                            )

                        return {"success": True, "voices": voices, "total": len(voices)}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("detail", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class SpotifyIntegration(BaseIntegration):
    """Spotify Web API integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
            "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
            "base_url": "https://api.spotify.com/v1",
        }
        super().__init__("spotify", {**default_config, **(config or {})})
        self.access_token = None

    async def initialize(self):
        """Initialize Spotify API"""
        if not self.config["client_id"] or not self.config["client_secret"]:
            logger.warning("⚠️  Spotify credentials not configured")
        else:
            await self._get_access_token()
            logger.info("✅ Spotify integration initialized")

    async def cleanup(self):
        """Cleanup Spotify resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Spotify API health"""
        if not self.access_token:
            return {"status": "unhealthy", "error": "Not authenticated"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.access_token}"}
                async with session.get(f"{self.config['base_url']}/browse/categories", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _get_access_token(self):
        """Get OAuth access token"""
        try:
            import base64

            credentials = base64.b64encode(
                f"{self.config['client_id']}:{self.config['client_secret']}".encode()
            ).decode()

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Basic {credentials}"}
                data = {"grant_type": "client_credentials"}

                async with session.post(
                    "https://accounts.spotify.com/api/token", headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data["access_token"]
        except Exception as e:
            logger.error(f"Failed to get Spotify access token: {e}")

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Spotify operations"""

        if operation == "search":
            return await self._search(**kwargs)
        elif operation == "get_track_features":
            return await self._get_track_features(**kwargs)
        elif operation == "create_playlist":
            return await self._create_playlist(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _search(self, query: str, search_type: str = "track", limit: int = 20) -> Dict[str, Any]:
        """Search Spotify catalog"""
        if not self.access_token:
            return {"success": False, "error": "Not authenticated"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.access_token}"}
                params = {"q": query, "type": search_type, "limit": limit}

                async with session.get(f"{self.config['base_url']}/search", headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"success": True, "query": query, "search_type": search_type, "results": data}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("error", {}).get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class PodcastIntegration(BaseIntegration):
    """Podcast RSS and management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"default_feed_limit": 50, "download_dir": os.path.join(tempfile.gettempdir(), "podcasts")}
        super().__init__("podcast", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize podcast integration"""
        os.makedirs(self.config["download_dir"], exist_ok=True)
        logger.info("✅ Podcast integration initialized")

    async def cleanup(self):
        """Cleanup podcast resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check podcast integration health"""
        return {"status": "healthy"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute podcast operations"""

        if operation == "parse_feed":
            return await self._parse_feed(**kwargs)
        elif operation == "download_episode":
            return await self._download_episode(**kwargs)
        elif operation == "search_podcasts":
            return await self._search_podcasts(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _parse_feed(self, feed_url: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """Parse podcast RSS feed"""
        try:
            import feedparser

            # Parse feed
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                return {"success": False, "error": "Invalid RSS feed"}

            episodes = []
            for entry in feed.entries[: limit or self.config["default_feed_limit"]]:
                episode = {
                    "title": entry.get("title", ""),
                    "description": entry.get("description", ""),
                    "published": entry.get("published", ""),
                    "duration": entry.get("itunes_duration", ""),
                    "link": entry.get("link", ""),
                    "enclosure": entry.get("enclosures", [{}])[0].get("href", "") if entry.get("enclosures") else "",
                }
                episodes.append(episode)

            return {
                "success": True,
                "podcast": {
                    "title": feed.feed.get("title", ""),
                    "description": feed.feed.get("description", ""),
                    "author": feed.feed.get("author", ""),
                    "image": feed.feed.get("image", {}).get("href", ""),
                    "language": feed.feed.get("language", ""),
                },
                "episodes": episodes,
                "total_episodes": len(episodes),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
