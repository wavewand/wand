"""
IoT and hardware integrations for Wand
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class ArduinoIntegration(BaseIntegration):
    """Arduino microcontroller integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "serial_port": os.getenv("ARDUINO_SERIAL_PORT", "/dev/ttyUSB0"),
            "baud_rate": int(os.getenv("ARDUINO_BAUD_RATE", "9600")),
            "timeout": 2.0,
        }
        super().__init__("arduino", {**default_config, **(config or {})})
        self.serial_connection = None

    async def initialize(self):
        """Initialize Arduino integration"""
        try:
            # Try to import pyserial
            import serial

            logger.info("✅ Arduino integration initialized (pyserial available)")
        except ImportError:
            logger.warning("⚠️  Arduino integration requires 'pyserial' package")

    async def cleanup(self):
        """Cleanup Arduino resources"""
        if self.serial_connection and hasattr(self.serial_connection, 'close'):
            self.serial_connection.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check Arduino connection health"""
        try:
            import serial

            # Try to open serial connection
            try:
                test_conn = serial.Serial(
                    self.config["serial_port"], self.config["baud_rate"], timeout=self.config["timeout"]
                )
                test_conn.close()
                return {
                    "status": "healthy",
                    "serial_port": self.config["serial_port"],
                    "baud_rate": self.config["baud_rate"],
                }
            except serial.SerialException:
                return {"status": "unhealthy", "error": f"Cannot connect to {self.config['serial_port']}"}

        except ImportError:
            return {"status": "unhealthy", "error": "pyserial package not installed"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Arduino operations"""

        if operation == "send_command":
            return await self._send_command(**kwargs)
        elif operation == "read_data":
            return await self._read_data(**kwargs)
        elif operation == "upload_sketch":
            return await self._upload_sketch(**kwargs)
        elif operation == "get_board_info":
            return await self._get_board_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _send_command(self, command: str, wait_response: bool = True) -> Dict[str, Any]:
        """Send command to Arduino"""
        try:
            import serial

            with serial.Serial(
                self.config["serial_port"], self.config["baud_rate"], timeout=self.config["timeout"]
            ) as conn:
                # Send command
                conn.write(f"{command}\n".encode())

                response = ""
                if wait_response:
                    response = conn.readline().decode().strip()

                return {
                    "success": True,
                    "command": command,
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except ImportError:
            return {"success": False, "error": "pyserial package not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class RaspberryPiIntegration(BaseIntegration):
    """Raspberry Pi single-board computer integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {"gpio_enabled": True, "spi_enabled": False, "i2c_enabled": False}
        super().__init__("raspberrypi", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Raspberry Pi integration"""
        try:
            # Try to import RPi.GPIO
            import RPi.GPIO as GPIO

            logger.info("✅ Raspberry Pi integration initialized (RPi.GPIO available)")
        except ImportError:
            logger.warning("⚠️  Raspberry Pi integration requires 'RPi.GPIO' package")

    async def cleanup(self):
        """Cleanup Raspberry Pi resources"""
        try:
            import RPi.GPIO as GPIO

            GPIO.cleanup()
        except ImportError:
            pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Raspberry Pi system health"""
        try:
            import RPi.GPIO as GPIO

            # Get system info
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()

                # Extract Pi model info
                model_line = [line for line in cpuinfo.split('\n') if 'Model' in line]
                model = model_line[0].split(':')[1].strip() if model_line else "Unknown"

                # Get temperature
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        temp_c = int(f.read()) / 1000.0
                except BaseException:
                    temp_c = None

                return {"status": "healthy", "model": model, "temperature_c": temp_c, "gpio_available": True}

            except Exception as e:
                return {"status": "partial", "error": str(e), "gpio_available": True}

        except ImportError:
            return {"status": "unhealthy", "error": "RPi.GPIO package not installed"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Raspberry Pi operations"""

        if operation == "set_gpio":
            return await self._set_gpio(**kwargs)
        elif operation == "read_gpio":
            return await self._read_gpio(**kwargs)
        elif operation == "get_system_info":
            return await self._get_system_info(**kwargs)
        elif operation == "control_pwm":
            return await self._control_pwm(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _set_gpio(self, pin: int, value: bool, mode: str = "out") -> Dict[str, Any]:
        """Set GPIO pin value"""
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT if mode == "out" else GPIO.IN)

            if mode == "out":
                GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)

            return {
                "success": True,
                "pin": pin,
                "value": value,
                "mode": mode,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            return {"success": False, "error": "RPi.GPIO package not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ESPIntegration(BaseIntegration):
    """ESP32/ESP8266 microcontroller integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "device_ip": os.getenv("ESP_DEVICE_IP", "192.168.1.100"),
            "api_port": int(os.getenv("ESP_API_PORT", "80")),
            "timeout": 5.0,
        }
        super().__init__("esp", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize ESP integration"""
        logger.info("✅ ESP integration initialized")

    async def cleanup(self):
        """Cleanup ESP resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check ESP device health"""
        import aiohttp

        try:
            device_url = f"http://{self.config['device_ip']}:{self.config['api_port']}"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['timeout'])) as session:
                async with session.get(f"{device_url}/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "device_ip": self.config["device_ip"],
                            "uptime": data.get("uptime"),
                            "free_heap": data.get("free_heap"),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"Device returned {response.status}"}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute ESP operations"""

        if operation == "send_command":
            return await self._send_command(**kwargs)
        elif operation == "read_sensors":
            return await self._read_sensors(**kwargs)
        elif operation == "control_gpio":
            return await self._control_gpio(**kwargs)
        elif operation == "get_device_info":
            return await self._get_device_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}


class MQTTIntegration(BaseIntegration):
    """MQTT message broker integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "broker_host": os.getenv("MQTT_BROKER_HOST", "localhost"),
            "broker_port": int(os.getenv("MQTT_BROKER_PORT", "1883")),
            "username": os.getenv("MQTT_USERNAME", ""),
            "password": os.getenv("MQTT_PASSWORD", ""),
            "client_id": os.getenv("MQTT_CLIENT_ID", "wand-mqtt-client"),
        }
        super().__init__("mqtt", {**default_config, **(config or {})})
        self.client = None

    async def initialize(self):
        """Initialize MQTT integration"""
        try:
            import paho.mqtt.client as mqtt_client

            logger.info("✅ MQTT integration initialized (paho-mqtt available)")
        except ImportError:
            logger.warning("⚠️  MQTT integration requires 'paho-mqtt' package")

    async def cleanup(self):
        """Cleanup MQTT resources"""
        if self.client:
            self.client.disconnect()

    async def health_check(self) -> Dict[str, Any]:
        """Check MQTT broker health"""
        try:
            import paho.mqtt.client as mqtt_client

            # Test connection to broker
            test_client = mqtt_client.Client(client_id=f"{self.config['client_id']}-test")

            if self.config["username"] and self.config["password"]:
                test_client.username_pw_set(self.config["username"], self.config["password"])

            try:
                test_client.connect(self.config["broker_host"], self.config["broker_port"], 60)
                test_client.disconnect()

                return {
                    "status": "healthy",
                    "broker": f"{self.config['broker_host']}:{self.config['broker_port']}",
                    "authenticated": bool(self.config["username"]),
                }
            except Exception as e:
                return {"status": "unhealthy", "error": f"Connection failed: {str(e)}"}

        except ImportError:
            return {"status": "unhealthy", "error": "paho-mqtt package not installed"}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute MQTT operations"""

        if operation == "publish":
            return await self._publish(**kwargs)
        elif operation == "subscribe":
            return await self._subscribe(**kwargs)
        elif operation == "get_broker_info":
            return await self._get_broker_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> Dict[str, Any]:
        """Publish message to MQTT topic"""
        try:
            import paho.mqtt.client as mqtt_client

            client = mqtt_client.Client(client_id=self.config["client_id"])

            if self.config["username"] and self.config["password"]:
                client.username_pw_set(self.config["username"], self.config["password"])

            client.connect(self.config["broker_host"], self.config["broker_port"], 60)

            result = client.publish(topic, payload, qos, retain)
            client.disconnect()

            return {
                "success": True,
                "topic": topic,
                "payload": payload,
                "message_id": result.mid,
                "qos": qos,
                "retain": retain,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            return {"success": False, "error": "paho-mqtt package not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
