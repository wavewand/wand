"""
Host Agent execution backend - execute commands via separate host agent service
"""

import asyncio
import json
import time
from typing import Any, Dict

import aiohttp

from .base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus


class HostAgentExecutionBackend(ExecutionBackend):
    """Execute commands via host agent service"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get('url', 'http://host-agent:8001')
        self.auth_token = config.get('auth_token', '')
        self.timeout = config.get('timeout', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.health_check_interval = config.get('health_check_interval', 60)

        # HTTP session
        self._session = None
        self._last_health_check = 0
        self._is_healthy = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            headers = {}
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers, timeout=timeout, connector=aiohttp.TCPConnector(limit=10)
            )

        return self._session

    async def execute(self, exec_config: ExecutionConfig) -> ExecutionResult:
        """Execute command via host agent"""
        start_time = time.time()

        # Check health if needed
        if not await self._ensure_healthy():
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                stderr="Host agent is not healthy",
                exit_code=1,
                command=exec_config.command,
                working_directory=exec_config.working_directory or "",
                execution_time=time.time() - start_time,
            )

        # Prepare request payload
        payload = {
            'command': exec_config.command,
            'working_directory': exec_config.working_directory,
            'timeout': exec_config.timeout,
            'environment': exec_config.environment or {},
            'input_data': exec_config.input_data,
            'capture_output': exec_config.capture_output,
            'shell': exec_config.shell,
            'user_id': exec_config.user_id,
            'metadata': exec_config.metadata or {},
        }

        # Execute with retry logic
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                session = await self._get_session()

                async with session.post(f'{self.url}/execute', json=payload) as response:
                    if response.status == 200:
                        result_data = await response.json()

                        # Convert response to ExecutionResult
                        result = ExecutionResult(
                            status=ExecutionStatus(result_data.get('status', 'failed')),
                            stdout=result_data.get('stdout', ''),
                            stderr=result_data.get('stderr', ''),
                            exit_code=result_data.get('exit_code', 1),
                            command=exec_config.command,
                            working_directory=result_data.get('working_directory', ''),
                            execution_time=result_data.get('execution_time', time.time() - start_time),
                            metadata=result_data.get('metadata', {}),
                        )

                        # Audit log
                        await self.audit_log(exec_config, result)

                        return result

                    elif response.status == 401:
                        error_text = await response.text()
                        return ExecutionResult(
                            status=ExecutionStatus.BLOCKED,
                            stderr=f"Authentication failed: {error_text}",
                            exit_code=1,
                            command=exec_config.command,
                            working_directory=exec_config.working_directory or "",
                            execution_time=time.time() - start_time,
                        )

                    elif response.status == 403:
                        error_text = await response.text()
                        return ExecutionResult(
                            status=ExecutionStatus.BLOCKED,
                            stderr=f"Command blocked by host agent: {error_text}",
                            exit_code=1,
                            command=exec_config.command,
                            working_directory=exec_config.working_directory or "",
                            execution_time=time.time() - start_time,
                        )

                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"

            except asyncio.TimeoutError:
                last_error = f"Request timed out after {self.timeout}s"

            except aiohttp.ClientConnectorError as e:
                last_error = f"Connection failed: {str(e)}"
                self._is_healthy = False

            except Exception as e:
                last_error = f"Request failed: {str(e)}"

            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)

        # All retries failed
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            stderr=f"Host agent execution failed after {self.retry_attempts} attempts. Last error: {last_error}",
            exit_code=1,
            command=exec_config.command,
            working_directory=exec_config.working_directory or "",
            execution_time=time.time() - start_time,
        )

    async def health_check(self) -> bool:
        """Check if host agent is healthy"""
        try:
            session = await self._get_session()

            async with session.get(f'{self.url}/health') as response:
                if response.status == 200:
                    health_data = await response.json()
                    return health_data.get('status') == 'healthy'

        except Exception as e:
            self.logger.error(f"Host agent health check failed: {e}")

        return False

    async def _ensure_healthy(self) -> bool:
        """Ensure host agent is healthy, checking if needed"""
        current_time = time.time()

        # Check if we need to perform health check
        if (current_time - self._last_health_check) > self.health_check_interval:
            self._is_healthy = await self.health_check()
            self._last_health_check = current_time

        return self._is_healthy

    async def cleanup(self):
        """Clean up host agent resources"""
        # Cancel any active executions
        for execution_id in list(self._active_executions.keys()):
            await self.cancel_execution(execution_id)

        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()

        self.logger.info("Host agent execution backend cleaned up")

    async def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the host agent"""
        try:
            session = await self._get_session()

            async with session.get(f'{self.url}/info') as response:
                if response.status == 200:
                    return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get host agent info: {e}")

        return {'error': 'Failed to retrieve host agent information', 'url': self.url, 'configured': True}

    async def list_capabilities(self) -> Dict[str, Any]:
        """List host agent capabilities"""
        try:
            session = await self._get_session()

            async with session.get(f'{self.url}/capabilities') as response:
                if response.status == 200:
                    return await response.json()

        except Exception as e:
            self.logger.error(f"Failed to get host agent capabilities: {e}")

        return {'error': 'Failed to retrieve capabilities', 'available': False}
