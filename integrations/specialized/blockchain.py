"""
Blockchain and Web3 integrations for Wand
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class EthereumIntegration(BaseIntegration):
    """Ethereum blockchain integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "rpc_url": os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/"),
            "infura_project_id": os.getenv("INFURA_PROJECT_ID", ""),
            "private_key": os.getenv("ETHEREUM_PRIVATE_KEY", ""),
            "network": os.getenv("ETHEREUM_NETWORK", "mainnet"),
        }
        super().__init__("ethereum", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Ethereum integration"""
        try:
            from web3 import Web3

            logger.info("✅ Ethereum integration initialized (web3.py available)")
        except ImportError:
            logger.warning("⚠️  Ethereum integration requires 'web3' package")

    async def cleanup(self):
        """Cleanup Ethereum resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Ethereum node health"""
        try:
            from web3 import Web3

            # Construct RPC URL
            rpc_url = self.config["rpc_url"]
            if self.config["infura_project_id"] and "infura.io" in rpc_url:
                rpc_url = f"{rpc_url}{self.config['infura_project_id']}"

            w3 = Web3(Web3.HTTPProvider(rpc_url))

            if w3.is_connected():
                latest_block = w3.eth.block_number
                return {
                    "status": "healthy",
                    "network": self.config["network"],
                    "latest_block": latest_block,
                    "node_connected": True,
                }
            else:
                return {"status": "unhealthy", "error": "Cannot connect to Ethereum node"}

        except ImportError:
            return {"status": "unhealthy", "error": "web3 package not installed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Ethereum operations"""

        if operation == "get_balance":
            return await self._get_balance(**kwargs)
        elif operation == "get_transaction":
            return await self._get_transaction(**kwargs)
        elif operation == "send_transaction":
            return await self._send_transaction(**kwargs)
        elif operation == "call_contract":
            return await self._call_contract(**kwargs)
        elif operation == "get_gas_price":
            return await self._get_gas_price(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_balance(self, address: str, block: str = "latest") -> Dict[str, Any]:
        """Get Ethereum balance for address"""
        try:
            from web3 import Web3

            rpc_url = self.config["rpc_url"]
            if self.config["infura_project_id"] and "infura.io" in rpc_url:
                rpc_url = f"{rpc_url}{self.config['infura_project_id']}"

            w3 = Web3(Web3.HTTPProvider(rpc_url))

            if not w3.is_connected():
                return {"success": False, "error": "Cannot connect to Ethereum node"}

            balance_wei = w3.eth.get_balance(Web3.to_checksum_address(address), block)
            balance_eth = w3.from_wei(balance_wei, 'ether')

            return {
                "success": True,
                "address": address,
                "balance_wei": str(balance_wei),
                "balance_eth": str(balance_eth),
                "block": block,
            }

        except ImportError:
            return {"success": False, "error": "web3 package not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class BitcoinIntegration(BaseIntegration):
    """Bitcoin blockchain integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "rpc_url": os.getenv("BITCOIN_RPC_URL", "https://blockstream.info/api"),
            "network": os.getenv("BITCOIN_NETWORK", "mainnet"),
            "api_key": os.getenv("BITCOIN_API_KEY", ""),
        }
        super().__init__("bitcoin", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Bitcoin integration"""
        logger.info("✅ Bitcoin integration initialized")

    async def cleanup(self):
        """Cleanup Bitcoin resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Bitcoin API health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['rpc_url']}/blocks/tip/height") as response:
                    if response.status == 200:
                        height = await response.text()
                        return {
                            "status": "healthy",
                            "network": self.config["network"],
                            "latest_block_height": int(height),
                        }
                    else:
                        return {"status": "unhealthy", "error": f"API returned {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Bitcoin operations"""

        if operation == "get_address_info":
            return await self._get_address_info(**kwargs)
        elif operation == "get_transaction":
            return await self._get_transaction(**kwargs)
        elif operation == "get_block":
            return await self._get_block(**kwargs)
        elif operation == "get_mempool_info":
            return await self._get_mempool_info(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_address_info(self, address: str) -> Dict[str, Any]:
        """Get Bitcoin address information"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['rpc_url']}/address/{address}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "address": address,
                            "balance": data.get("chain_stats", {}).get("funded_txo_sum", 0),
                            "tx_count": data.get("chain_stats", {}).get("tx_count", 0),
                            "received": data.get("chain_stats", {}).get("funded_txo_sum", 0),
                            "spent": data.get("chain_stats", {}).get("spent_txo_sum", 0),
                        }
                    else:
                        return {"success": False, "error": f"API returned {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class Web3Integration(BaseIntegration):
    """General Web3 and DeFi integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "default_chain": os.getenv("WEB3_DEFAULT_CHAIN", "ethereum"),
            "chains": {
                "ethereum": {"rpc_url": os.getenv("ETHEREUM_RPC_URL", ""), "chain_id": 1},
                "polygon": {"rpc_url": os.getenv("POLYGON_RPC_URL", ""), "chain_id": 137},
                "bsc": {"rpc_url": os.getenv("BSC_RPC_URL", ""), "chain_id": 56},
            },
        }
        super().__init__("web3", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Web3 integration"""
        try:
            from web3 import Web3

            logger.info("✅ Web3 integration initialized")
        except ImportError:
            logger.warning("⚠️  Web3 integration requires 'web3' package")

    async def cleanup(self):
        """Cleanup Web3 resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Web3 connections health"""
        try:
            from web3 import Web3

            status = {"status": "healthy", "chains": {}}

            for chain_name, chain_config in self.config["chains"].items():
                if chain_config["rpc_url"]:
                    try:
                        w3 = Web3(Web3.HTTPProvider(chain_config["rpc_url"]))
                        if w3.is_connected():
                            status["chains"][chain_name] = {
                                "connected": True,
                                "latest_block": w3.eth.block_number,
                                "chain_id": chain_config["chain_id"],
                            }
                        else:
                            status["chains"][chain_name] = {"connected": False}
                    except BaseException:
                        status["chains"][chain_name] = {"connected": False, "error": "Connection failed"}
                else:
                    status["chains"][chain_name] = {"configured": False}

            return status

        except ImportError:
            return {"status": "unhealthy", "error": "web3 package not installed"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Web3 operations"""

        if operation == "get_token_balance":
            return await self._get_token_balance(**kwargs)
        elif operation == "swap_tokens":
            return await self._swap_tokens(**kwargs)
        elif operation == "get_defi_rates":
            return await self._get_defi_rates(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}


class NFTIntegration(BaseIntegration):
    """NFT and digital collectibles integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "opensea_api_key": os.getenv("OPENSEA_API_KEY", ""),
            "alchemy_api_key": os.getenv("ALCHEMY_API_KEY", ""),
            "moralis_api_key": os.getenv("MORALIS_API_KEY", ""),
        }
        super().__init__("nft", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize NFT integration"""
        logger.info("✅ NFT integration initialized")

    async def cleanup(self):
        """Cleanup NFT resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check NFT service health"""
        status = {"status": "healthy", "services": {}}

        # Check OpenSea API
        if self.config["opensea_api_key"]:
            try:
                headers = {"X-API-KEY": self.config["opensea_api_key"]}
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.opensea.io/api/v1/collections", headers=headers, params={"limit": 1}
                    ) as response:
                        status["services"]["opensea"] = {"connected": response.status == 200}
            except BaseException:
                status["services"]["opensea"] = {"connected": False}
        else:
            status["services"]["opensea"] = {"configured": False}

        return status

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute NFT operations"""

        if operation == "get_nft_collection":
            return await self._get_nft_collection(**kwargs)
        elif operation == "get_nft_metadata":
            return await self._get_nft_metadata(**kwargs)
        elif operation == "list_user_nfts":
            return await self._list_user_nfts(**kwargs)
        elif operation == "get_floor_price":
            return await self._get_floor_price(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_nft_collection(self, collection_slug: str) -> Dict[str, Any]:
        """Get NFT collection information"""
        if not self.config["opensea_api_key"]:
            return {"success": False, "error": "OpenSea API key not configured"}

        try:
            headers = {"X-API-KEY": self.config["opensea_api_key"]}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.opensea.io/api/v1/collection/{collection_slug}", headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        collection = data.get("collection", {})

                        return {
                            "success": True,
                            "name": collection.get("name"),
                            "description": collection.get("description"),
                            "image_url": collection.get("image_url"),
                            "floor_price": collection.get("stats", {}).get("floor_price"),
                            "total_supply": collection.get("stats", {}).get("total_supply"),
                            "owners": collection.get("stats", {}).get("num_owners"),
                            "volume_traded": collection.get("stats", {}).get("total_volume"),
                        }
                    else:
                        return {"success": False, "error": f"OpenSea API returned {response.status}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
