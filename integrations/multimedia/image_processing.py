"""
Image processing integrations for Wand
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Optional imports for image processing
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = ImageDraw = ImageFont = ImageFilter = None

try:
    import qrcode

    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
    qrcode = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = sns = None

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class ImageIntegration(BaseIntegration):
    """General image processing integration using PIL"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is not installed. Install with: pip install Pillow")
        default_config = {
            "max_image_size": (4096, 4096),
            "default_quality": 85,
            "supported_formats": ["JPEG", "PNG", "WEBP", "BMP", "TIFF"],
        }
        super().__init__("image", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize image processing"""
        logger.info("✅ Image processing initialized successfully")

    async def cleanup(self):
        """Cleanup image resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check image processing health"""
        return {"status": "healthy", "pil_available": True}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute image operations"""

        if operation == "resize":
            return await self._resize_image(**kwargs)
        elif operation == "crop":
            return await self._crop_image(**kwargs)
        elif operation == "rotate":
            return await self._rotate_image(**kwargs)
        elif operation == "filter":
            return await self._apply_filter(**kwargs)
        elif operation == "convert_format":
            return await self._convert_format(**kwargs)
        elif operation == "add_text":
            return await self._add_text(**kwargs)
        elif operation == "merge":
            return await self._merge_images(**kwargs)
        elif operation == "get_info":
            return await self._get_image_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _resize_image(
        self, input_path: str, output_path: str, width: int, height: int, maintain_aspect: bool = True
    ) -> Dict[str, Any]:
        """Resize image"""
        try:
            with Image.open(input_path) as img:
                if maintain_aspect:
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                else:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)

                img.save(output_path, quality=self.config["default_quality"])

                return {
                    "success": True,
                    "output_path": output_path,
                    "original_size": f"{img.width}x{img.height}",
                    "new_size": f"{width}x{height}",
                    "maintain_aspect": maintain_aspect,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _crop_image(
        self, input_path: str, output_path: str, left: int, top: int, right: int, bottom: int
    ) -> Dict[str, Any]:
        """Crop image"""
        try:
            with Image.open(input_path) as img:
                cropped = img.crop((left, top, right, bottom))
                cropped.save(output_path, quality=self.config["default_quality"])

                return {
                    "success": True,
                    "output_path": output_path,
                    "crop_box": [left, top, right, bottom],
                    "cropped_size": f"{cropped.width}x{cropped.height}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _rotate_image(self, input_path: str, output_path: str, angle: float) -> Dict[str, Any]:
        """Rotate image"""
        try:
            with Image.open(input_path) as img:
                rotated = img.rotate(angle, expand=True)
                rotated.save(output_path, quality=self.config["default_quality"])

                return {
                    "success": True,
                    "output_path": output_path,
                    "angle": angle,
                    "new_size": f"{rotated.width}x{rotated.height}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_filter(self, input_path: str, output_path: str, filter_type: str) -> Dict[str, Any]:
        """Apply filter to image"""
        try:
            filter_map = {
                "blur": ImageFilter.BLUR,
                "contour": ImageFilter.CONTOUR,
                "detail": ImageFilter.DETAIL,
                "edge_enhance": ImageFilter.EDGE_ENHANCE,
                "emboss": ImageFilter.EMBOSS,
                "sharpen": ImageFilter.SHARPEN,
                "smooth": ImageFilter.SMOOTH,
            }

            if filter_type not in filter_map:
                return {"success": False, "error": f"Unknown filter: {filter_type}"}

            with Image.open(input_path) as img:
                filtered = img.filter(filter_map[filter_type])
                filtered.save(output_path, quality=self.config["default_quality"])

                return {"success": True, "output_path": output_path, "filter": filter_type}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_image_info(self, input_path: str) -> Dict[str, Any]:
        """Get image information"""
        try:
            with Image.open(input_path) as img:
                return {
                    "success": True,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_bytes": os.path.getsize(input_path),
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


class OCRIntegration(BaseIntegration):
    """OCR (Optical Character Recognition) integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # OCR requires external tesseract binary, not a Python package
        # So we don't check for imports here, just configuration
        default_config = {"languages": ["eng"], "confidence_threshold": 60}  # Default language
        super().__init__("ocr", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize OCR"""
        try:
            import pytesseract

            logger.info("✅ OCR (Tesseract) initialized successfully")
        except ImportError:
            logger.warning("⚠️  Tesseract not available - OCR functionality limited")

    async def cleanup(self):
        """Cleanup OCR resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check OCR health"""
        try:
            import pytesseract

            return {"status": "healthy", "tesseract_available": True}
        except ImportError:
            return {"status": "unhealthy", "error": "Tesseract not installed"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute OCR operations"""

        if operation == "extract_text":
            return await self._extract_text(**kwargs)
        elif operation == "detect_text_boxes":
            return await self._detect_text_boxes(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _extract_text(self, image_path: str, language: str = "eng") -> Dict[str, Any]:
        """Extract text from image"""
        try:
            import pytesseract
            from PIL import Image

            with Image.open(image_path) as img:
                # Extract text
                text = pytesseract.image_to_string(img, lang=language)

                # Get confidence data
                data = pytesseract.image_to_data(img, lang=language, output_type=pytesseract.Output.DICT)

                # Filter out low confidence text
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                return {
                    "success": True,
                    "text": text.strip(),
                    "language": language,
                    "average_confidence": avg_confidence,
                    "word_count": len(text.split()),
                    "character_count": len(text),
                }

        except ImportError:
            return {"success": False, "error": "Tesseract not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class QRIntegration(BaseIntegration):
    """QR code and barcode integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"default_size": 10, "default_border": 4, "error_correction": "M"}  # L, M, Q, H
        super().__init__("qr", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize QR code processing"""
        logger.info("✅ QR code processing initialized successfully")

    async def cleanup(self):
        """Cleanup QR resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check QR processing health"""
        return {"status": "healthy", "qrcode_available": True}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute QR operations"""

        if operation == "generate":
            return await self._generate_qr(**kwargs)
        elif operation == "decode":
            return await self._decode_qr(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _generate_qr(self, data: str, output_path: str, size: Optional[int] = None) -> Dict[str, Any]:
        """Generate QR code"""
        try:
            # Error correction mapping
            error_correction_map = {
                "L": qrcode.constants.ERROR_CORRECT_L,
                "M": qrcode.constants.ERROR_CORRECT_M,
                "Q": qrcode.constants.ERROR_CORRECT_Q,
                "H": qrcode.constants.ERROR_CORRECT_H,
            }

            qr = qrcode.QRCode(
                version=1,
                error_correction=error_correction_map.get(
                    self.config["error_correction"], qrcode.constants.ERROR_CORRECT_M
                ),
                box_size=size or self.config["default_size"],
                border=self.config["default_border"],
            )

            qr.add_data(data)
            qr.make(fit=True)

            # Create QR code image
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(output_path)

            return {
                "success": True,
                "output_path": output_path,
                "data": data,
                "size": size or self.config["default_size"],
                "error_correction": self.config["error_correction"],
                "image_size": f"{img.width}x{img.height}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _decode_qr(self, image_path: str) -> Dict[str, Any]:
        """Decode QR code from image"""
        try:
            from PIL import Image
            from pyzbar import pyzbar

            with Image.open(image_path) as img:
                # Decode all barcodes/QR codes in image
                decoded_objects = pyzbar.decode(img)

                results = []
                for obj in decoded_objects:
                    results.append(
                        {
                            "type": obj.type,
                            "data": obj.data.decode('utf-8'),
                            "rect": {
                                "left": obj.rect.left,
                                "top": obj.rect.top,
                                "width": obj.rect.width,
                                "height": obj.rect.height,
                            },
                        }
                    )

                return {"success": True, "codes_found": len(results), "codes": results}

        except ImportError:
            return {"success": False, "error": "pyzbar not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ChartIntegration(BaseIntegration):
    """Chart and data visualization integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is not installed. Install with: pip install matplotlib seaborn")
        default_config = {"default_style": "whitegrid", "default_palette": "deep", "figure_size": (10, 6), "dpi": 300}
        super().__init__("chart", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize chart processing"""
        # Set default style
        sns.set_style(self.config["default_style"])
        sns.set_palette(self.config["default_palette"])
        logger.info("✅ Chart generation initialized successfully")

    async def cleanup(self):
        """Cleanup chart resources"""
        plt.close('all')

    async def health_check(self) -> Dict[str, Any]:
        """Check chart processing health"""
        return {"status": "healthy", "matplotlib_available": True, "seaborn_available": True}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute chart operations"""

        if operation == "bar_chart":
            return await self._create_bar_chart(**kwargs)
        elif operation == "line_chart":
            return await self._create_line_chart(**kwargs)
        elif operation == "pie_chart":
            return await self._create_pie_chart(**kwargs)
        elif operation == "scatter_plot":
            return await self._create_scatter_plot(**kwargs)
        elif operation == "histogram":
            return await self._create_histogram(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _create_bar_chart(
        self,
        data: Dict[str, Any],
        output_path: str,
        title: str = "Bar Chart",
        xlabel: str = "Categories",
        ylabel: str = "Values",
    ) -> Dict[str, Any]:
        """Create bar chart"""
        try:
            plt.figure(figsize=self.config["figure_size"])

            categories = list(data.keys())
            values = list(data.values())

            plt.bar(categories, values)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
            plt.close()

            return {
                "success": True,
                "output_path": output_path,
                "chart_type": "bar",
                "title": title,
                "data_points": len(categories),
            }

        except Exception as e:
            plt.close()
            return {"success": False, "error": str(e)}

    async def _create_line_chart(
        self,
        x_data: List,
        y_data: List,
        output_path: str,
        title: str = "Line Chart",
        xlabel: str = "X",
        ylabel: str = "Y",
    ) -> Dict[str, Any]:
        """Create line chart"""
        try:
            plt.figure(figsize=self.config["figure_size"])

            plt.plot(x_data, y_data, marker='o')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
            plt.close()

            return {
                "success": True,
                "output_path": output_path,
                "chart_type": "line",
                "title": title,
                "data_points": len(x_data),
            }

        except Exception as e:
            plt.close()
            return {"success": False, "error": str(e)}

    async def _create_pie_chart(
        self, data: Dict[str, Any], output_path: str, title: str = "Pie Chart"
    ) -> Dict[str, Any]:
        """Create pie chart"""
        try:
            plt.figure(figsize=self.config["figure_size"])

            labels = list(data.keys())
            sizes = list(data.values())

            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title(title)
            plt.axis('equal')
            plt.tight_layout()

            plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
            plt.close()

            return {
                "success": True,
                "output_path": output_path,
                "chart_type": "pie",
                "title": title,
                "segments": len(labels),
            }

        except Exception as e:
            plt.close()
            return {"success": False, "error": str(e)}
