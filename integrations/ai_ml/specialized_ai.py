"""
Specialized AI integrations for Wand
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class StabilityIntegration(BaseIntegration):
    """Stability AI (Stable Diffusion) integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("STABILITY_API_KEY", ""),
            "base_url": "https://api.stability.ai/v1",
            "default_engine": "stable-diffusion-xl-1024-v1-0",
            "default_steps": 30,
            "default_cfg_scale": 7,
        }
        super().__init__("stability", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Stability AI integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  Stability AI API key not configured")
        else:
            logger.info("✅ Stability AI integration initialized")

    async def cleanup(self):
        """Cleanup Stability AI resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Stability AI API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            headers = {"Authorization": f"Bearer {self.config['api_key']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/engines/list", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Stability AI operations"""

        if operation == "text_to_image":
            return await self._text_to_image(**kwargs)
        elif operation == "image_to_image":
            return await self._image_to_image(**kwargs)
        elif operation == "upscale":
            return await self._upscale_image(**kwargs)
        elif operation == "list_engines":
            return await self._list_engines(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _text_to_image(
        self,
        prompt: str,
        output_path: str = None,
        engine: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        samples: int = 1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate image from text prompt"""
        if not self.config["api_key"]:
            return {"success": False, "error": "Stability AI API key not configured"}

        engine = engine or self.config["default_engine"]
        steps = steps or self.config["default_steps"]
        cfg_scale = cfg_scale or self.config["default_cfg_scale"]

        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        body = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": samples,
            "steps": steps,
        }

        if seed is not None:
            body["seed"] = seed

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/generation/{engine}/text-to-image", headers=headers, json=body
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        images = []
                        for i, image in enumerate(data["artifacts"]):
                            # Decode base64 image
                            import base64

                            image_data = base64.b64decode(image["base64"])

                            # Save image
                            if not output_path:
                                import tempfile

                                output_path = os.path.join(
                                    tempfile.gettempdir(), f"stability_{int(datetime.now().timestamp())}_{i}.png"
                                )

                            with open(output_path, 'wb') as f:
                                f.write(image_data)

                            images.append(
                                {
                                    "path": output_path,
                                    "seed": image.get("seed"),
                                    "finish_reason": image.get("finishReason"),
                                }
                            )

                        return {
                            "success": True,
                            "images": images,
                            "prompt": prompt,
                            "engine": engine,
                            "parameters": {
                                "steps": steps,
                                "cfg_scale": cfg_scale,
                                "width": width,
                                "height": height,
                                "samples": samples,
                            },
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _image_to_image(
        self,
        init_image_path: str,
        prompt: str,
        output_path: str = None,
        engine: Optional[str] = None,
        image_strength: float = 0.35,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate image from image and text prompt"""
        if not self.config["api_key"]:
            return {"success": False, "error": "Stability AI API key not configured"}

        engine = engine or self.config["default_engine"]
        steps = steps or self.config["default_steps"]
        cfg_scale = cfg_scale or self.config["default_cfg_scale"]

        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Accept": "application/json"}

        # Prepare multipart form data
        data = aiohttp.FormData()
        data.add_field('text_prompts[0][text]', prompt)
        data.add_field('cfg_scale', str(cfg_scale))
        data.add_field('image_strength', str(image_strength))
        data.add_field('steps', str(steps))
        data.add_field('samples', '1')

        # Add image file
        with open(init_image_path, 'rb') as f:
            data.add_field('init_image', f, filename='init_image.png', content_type='image/png')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/generation/{engine}/image-to-image", headers=headers, data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Process first image
                        image = result["artifacts"][0]

                        # Decode base64 image
                        import base64

                        image_data = base64.b64decode(image["base64"])

                        # Save image
                        if not output_path:
                            import tempfile

                            output_path = os.path.join(
                                tempfile.gettempdir(), f"stability_img2img_{int(datetime.now().timestamp())}.png"
                            )

                        with open(output_path, 'wb') as f:
                            f.write(image_data)

                        return {
                            "success": True,
                            "output_path": output_path,
                            "prompt": prompt,
                            "init_image": init_image_path,
                            "engine": engine,
                            "parameters": {"image_strength": image_strength, "steps": steps, "cfg_scale": cfg_scale},
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_engines(self) -> Dict[str, Any]:
        """List available Stability AI engines"""
        if not self.config["api_key"]:
            return {"success": False, "error": "Stability AI API key not configured"}

        headers = {"Authorization": f"Bearer {self.config['api_key']}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/engines/list", headers=headers) as response:
                    if response.status == 200:
                        engines = await response.json()

                        engine_list = []
                        for engine in engines:
                            engine_list.append(
                                {
                                    "id": engine["id"],
                                    "name": engine["name"],
                                    "description": engine.get("description", ""),
                                    "type": engine.get("type", ""),
                                }
                            )

                        return {"success": True, "engines": engine_list, "total": len(engine_list)}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}


class DeepLIntegration(BaseIntegration):
    """DeepL translation integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "api_key": os.getenv("DEEPL_API_KEY", ""),
            "base_url": "https://api-free.deepl.com/v2",  # Use api.deepl.com for pro
            "default_target_lang": "EN",
            "preserve_formatting": True,
        }
        super().__init__("deepl", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize DeepL integration"""
        if not self.config["api_key"]:
            logger.warning("⚠️  DeepL API key not configured")
        else:
            logger.info("✅ DeepL integration initialized")

    async def cleanup(self):
        """Cleanup DeepL resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check DeepL API health"""
        if not self.config["api_key"]:
            return {"status": "unhealthy", "error": "API key not configured"}

        try:
            headers = {"Authorization": f"DeepL-Auth-Key {self.config['api_key']}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/usage", headers=headers) as response:
                    if response.status == 200:
                        return {"status": "healthy"}
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute DeepL operations"""

        if operation == "translate":
            return await self._translate_text(**kwargs)
        elif operation == "detect_language":
            return await self._detect_language(**kwargs)
        elif operation == "get_usage":
            return await self._get_usage(**kwargs)
        elif operation == "list_languages":
            return await self._list_languages(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _translate_text(
        self,
        text: str,
        target_lang: str = None,
        source_lang: Optional[str] = None,
        preserve_formatting: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Translate text"""
        if not self.config["api_key"]:
            return {"success": False, "error": "DeepL API key not configured"}

        target_lang = target_lang or self.config["default_target_lang"]
        preserve_formatting = (
            preserve_formatting if preserve_formatting is not None else self.config["preserve_formatting"]
        )

        headers = {"Authorization": f"DeepL-Auth-Key {self.config['api_key']}"}

        data = {"text": text, "target_lang": target_lang, "preserve_formatting": "1" if preserve_formatting else "0"}

        if source_lang:
            data["source_lang"] = source_lang

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.config['base_url']}/translate", headers=headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()

                        translation = result["translations"][0]

                        return {
                            "success": True,
                            "translated_text": translation["text"],
                            "detected_source_language": translation.get("detected_source_language"),
                            "target_language": target_lang,
                            "original_text": text,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        if not self.config["api_key"]:
            return {"success": False, "error": "DeepL API key not configured"}

        headers = {"Authorization": f"DeepL-Auth-Key {self.config['api_key']}"}
        data = {"text": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config['base_url']}/translate",
                    headers=headers,
                    data={**data, "target_lang": "EN"},  # Dummy translation to detect language
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        detected_lang = result["translations"][0].get("detected_source_language")

                        return {"success": True, "detected_language": detected_lang, "text": text}
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_usage(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        if not self.config["api_key"]:
            return {"success": False, "error": "DeepL API key not configured"}

        headers = {"Authorization": f"DeepL-Auth-Key {self.config['api_key']}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['base_url']}/usage", headers=headers) as response:
                    if response.status == 200:
                        usage = await response.json()

                        return {
                            "success": True,
                            "character_count": usage.get("character_count", 0),
                            "character_limit": usage.get("character_limit", 0),
                            "usage_percentage": (usage.get("character_count", 0) / usage.get("character_limit", 1))
                            * 100,
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _list_languages(self, lang_type: str = "target") -> Dict[str, Any]:
        """List supported languages"""
        if not self.config["api_key"]:
            return {"success": False, "error": "DeepL API key not configured"}

        headers = {"Authorization": f"DeepL-Auth-Key {self.config['api_key']}"}
        params = {"type": lang_type}  # "source" or "target"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/languages", headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        languages = await response.json()

                        language_list = []
                        for lang in languages:
                            language_list.append({"language": lang["language"], "name": lang["name"]})

                        return {
                            "success": True,
                            "languages": language_list,
                            "type": lang_type,
                            "total": len(language_list),
                        }
                    else:
                        error = await response.json()
                        return {"success": False, "error": error.get("message", "API error")}

        except Exception as e:
            return {"success": False, "error": str(e)}
