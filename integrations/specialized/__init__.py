"""
🎮 Specialized & Fun Integrations

Gaming, IoT, blockchain, and other specialized tools for Wand
"""

from .blockchain import BitcoinIntegration, EthereumIntegration, NFTIntegration, Web3Integration
from .gaming import DiscordBotIntegration, SteamIntegration, TwitchIntegration
from .iot_hardware import ArduinoIntegration, ESPIntegration, MQTTIntegration, RaspberryPiIntegration

# Initialize integration instances
steam_integration = SteamIntegration()
twitch_integration = TwitchIntegration()
discord_bot_integration = DiscordBotIntegration()

arduino_integration = ArduinoIntegration()
raspberrypi_integration = RaspberryPiIntegration()
esp_integration = ESPIntegration()
mqtt_integration = MQTTIntegration()

ethereum_integration = EthereumIntegration()
bitcoin_integration = BitcoinIntegration()
web3_integration = Web3Integration()
nft_integration = NFTIntegration()

__all__ = [
    # Gaming
    "steam_integration",
    "twitch_integration",
    "discord_bot_integration",
    # IoT & Hardware
    "arduino_integration",
    "raspberrypi_integration",
    "esp_integration",
    "mqtt_integration",
    # Blockchain & Web3
    "ethereum_integration",
    "bitcoin_integration",
    "web3_integration",
    "nft_integration",
]
