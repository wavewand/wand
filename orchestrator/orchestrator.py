"""
MCP System Process Orchestrator

Manages the startup, monitoring, and shutdown of all system processes.
"""

import asyncio
import logging
import multiprocessing as mp
import signal
import sys
import time
from typing import Dict, List, Optional

import psutil

from distributed.types import AgentType
from grpc_services.agent_service import run_agent_process
from grpc_services.coordinator_service import run_coordinator_process
from grpc_services.integration_service import run_integration_process
from rest_api.gateway import run_rest_api_process


class MCPSystemOrchestrator:
    """Orchestrates all MCP system processes."""

    def __init__(self):
        self.processes: Dict[str, mp.Process] = {}
        self.ports = {
            'coordinator': 50051,
            'integration': 50200,
            'rest_api': 8000,
            'agents': {
                AgentType.MANAGER: 50100,
                AgentType.FRONTEND: 50101,
                AgentType.BACKEND: 50102,
                AgentType.DATABASE: 50103,
                AgentType.DEVOPS: 50104,
                AgentType.INTEGRATION: 50105,
                AgentType.QA: 50106,
                AgentType.HAYSTACK: 50107,
                AgentType.LLAMAINDEX: 50108,
            },
        }

        self.logger = logging.getLogger("orchestrator")
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True

    def start_system(self):
        """Start the entire MCP system."""
        self.logger.info("Starting MCP Distributed System...")

        try:
            # Start processes in order
            self._start_coordinator()
            self._wait_for_service('coordinator', 5)

            self._start_integration_service()
            self._wait_for_service('integration', 5)

            self._start_agent_processes()
            self._wait_for_agents(10)

            self._start_rest_api()
            self._wait_for_service('rest_api', 5)

            self.logger.info("All services started successfully!")
            self._print_service_urls()

            # Monitor processes
            self._monitor_processes()

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.shutdown_system()
            raise
        finally:
            self.shutdown_system()

    def _start_coordinator(self):
        """Start coordinator gRPC service."""
        self.logger.info("Starting Coordinator service...")

        process = mp.Process(target=run_coordinator_process, args=(self.ports['coordinator'],), name="coordinator")
        process.start()
        self.processes['coordinator'] = process

        self.logger.info(f"Coordinator started on port {self.ports['coordinator']} (PID: {process.pid})")

    def _start_integration_service(self):
        """Start integration gRPC service."""
        self.logger.info("Starting Integration service...")

        process = mp.Process(target=run_integration_process, args=(self.ports['integration'],), name="integration")
        process.start()
        self.processes['integration'] = process

        self.logger.info(f"Integration service started on port {self.ports['integration']} (PID: {process.pid})")

    def _start_agent_processes(self):
        """Start all agent gRPC services."""
        self.logger.info("Starting Agent processes...")

        for agent_type, port in self.ports['agents'].items():
            process_name = f"agent_{agent_type.value}"

            process = mp.Process(target=run_agent_process, args=(agent_type, port), name=process_name)
            process.start()
            self.processes[process_name] = process

            self.logger.info(f"{agent_type.value.title()} agent started on port {port} (PID: {process.pid})")

    def _start_rest_api(self):
        """Start REST API gateway."""
        self.logger.info("Starting REST API Gateway...")

        process = mp.Process(
            target=run_rest_api_process,
            args=("0.0.0.0", self.ports['rest_api'], self.ports['coordinator'], self.ports['integration']),
            name="rest_api",
        )
        process.start()
        self.processes['rest_api'] = process

        self.logger.info(f"REST API Gateway started on port {self.ports['rest_api']} (PID: {process.pid})")

    def _wait_for_service(self, service_name: str, timeout: int):
        """Wait for a service to be ready."""
        self.logger.info(f"Waiting for {service_name} to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if service_name in self.processes:
                process = self.processes[service_name]
                if process.is_alive():
                    # Additional check could be added here to test actual connectivity
                    time.sleep(1)  # Give service time to initialize
                    return
            time.sleep(0.5)

        self.logger.warning(f"{service_name} may not be fully ready after {timeout}s")

    def _wait_for_agents(self, timeout: int):
        """Wait for all agents to be ready."""
        self.logger.info("Waiting for agents to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True
            for agent_type in self.ports['agents']:
                process_name = f"agent_{agent_type.value}"
                if process_name in self.processes:
                    process = self.processes[process_name]
                    if not process.is_alive():
                        all_ready = False
                        break
                else:
                    all_ready = False
                    break

            if all_ready:
                time.sleep(2)  # Give agents time to register with coordinator
                return

            time.sleep(0.5)

        self.logger.warning(f"Some agents may not be ready after {timeout}s")

    def _print_service_urls(self):
        """Print service URLs for easy access."""
        print("\n" + "=" * 60)
        print("MCP DISTRIBUTED SYSTEM - SERVICE ENDPOINTS")
        print("=" * 60)
        print(f"REST API:        http://localhost:{self.ports['rest_api']}")
        print(f"API Docs:        http://localhost:{self.ports['rest_api']}/docs")
        print(f"Health Check:    http://localhost:{self.ports['rest_api']}/health")
        print(f"System Status:   http://localhost:{self.ports['rest_api']}/api/v1/system/status")
        print("")
        print("gRPC Services:")
        print(f"  Coordinator:   127.0.0.1:{self.ports['coordinator']}")
        print(f"  Integration:   127.0.0.1:{self.ports['integration']}")
        print("")
        print("Agent Services:")
        for agent_type, port in self.ports['agents'].items():
            print(f"  {agent_type.value.title():12} 127.0.0.1:{port}")
        print("=" * 60)
        print("Press Ctrl+C to shutdown the system")
        print("=" * 60 + "\n")

    def _monitor_processes(self):
        """Monitor all processes and restart if needed."""
        self.logger.info("Starting process monitoring...")

        check_interval = 5  # seconds
        restart_attempts = {}
        max_restarts = 3

        while not self.shutdown_requested:
            try:
                # Check each process
                for process_name, process in list(self.processes.items()):
                    if not process.is_alive():
                        self.logger.warning(f"Process {process_name} (PID: {process.pid}) has died")

                        # Track restart attempts
                        if process_name not in restart_attempts:
                            restart_attempts[process_name] = 0

                        if restart_attempts[process_name] < max_restarts:
                            restart_attempts[process_name] += 1
                            self.logger.info(
                                f"Restarting {process_name} (attempt {restart_attempts[process_name]}/{max_restarts})"
                            )

                            # Restart process
                            self._restart_process(process_name)
                        else:
                            self.logger.error(f"Process {process_name} failed too many times, giving up")
                            # Remove from monitoring
                            del self.processes[process_name]

                # Sleep before next check
                time.sleep(check_interval)

            except KeyboardInterrupt:
                self.logger.info("Process monitoring interrupted")
                break
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                time.sleep(check_interval)

    def _restart_process(self, process_name: str):
        """Restart a specific process."""
        try:
            # Clean up old process
            if process_name in self.processes:
                old_process = self.processes[process_name]
                if old_process.is_alive():
                    old_process.terminate()
                    old_process.join(timeout=5)
                del self.processes[process_name]

            # Start new process based on type
            if process_name == 'coordinator':
                self._start_coordinator()
            elif process_name == 'integration':
                self._start_integration_service()
            elif process_name == 'rest_api':
                self._start_rest_api()
            elif process_name.startswith('agent_'):
                agent_type_name = process_name.replace('agent_', '')
                for agent_type in AgentType:
                    if agent_type.value == agent_type_name:
                        port = self.ports['agents'][agent_type]
                        process = mp.Process(target=run_agent_process, args=(agent_type, port), name=process_name)
                        process.start()
                        self.processes[process_name] = process
                        break

            self.logger.info(f"Successfully restarted {process_name}")

        except Exception as e:
            self.logger.error(f"Failed to restart {process_name}: {e}")

    def shutdown_system(self):
        """Shutdown all processes gracefully."""
        if not self.processes:
            return

        self.logger.info("Shutting down MCP Distributed System...")

        # Shutdown order: REST API -> Agents -> Integration -> Coordinator
        shutdown_order = [
            'rest_api',
            *[f'agent_{agent_type.value}' for agent_type in AgentType],
            'integration',
            'coordinator',
        ]

        for process_name in shutdown_order:
            if process_name in self.processes:
                self._shutdown_process(process_name)

        # Wait for all processes to terminate
        self.logger.info("Waiting for all processes to terminate...")
        for process_name, process in self.processes.items():
            try:
                process.join(timeout=10)
                if process.is_alive():
                    self.logger.warning(f"Force killing {process_name}")
                    process.kill()
                    process.join()
            except Exception as e:
                self.logger.error(f"Error shutting down {process_name}: {e}")

        self.processes.clear()
        self.logger.info("MCP Distributed System shutdown complete")

    def _shutdown_process(self, process_name: str):
        """Shutdown a specific process gracefully."""
        if process_name not in self.processes:
            return

        process = self.processes[process_name]

        if not process.is_alive():
            self.logger.info(f"Process {process_name} already terminated")
            return

        self.logger.info(f"Shutting down {process_name} (PID: {process.pid})")

        try:
            # Try graceful shutdown first
            process.terminate()
            process.join(timeout=5)

            if process.is_alive():
                self.logger.warning(f"Force killing {process_name}")
                process.kill()
                process.join(timeout=5)

            self.logger.info(f"Successfully shutdown {process_name}")

        except Exception as e:
            self.logger.error(f"Error shutting down {process_name}: {e}")

    def get_system_status(self) -> Dict:
        """Get current system status."""
        status = {
            'total_processes': len(self.processes),
            'running_processes': 0,
            'failed_processes': 0,
            'processes': {},
        }

        for process_name, process in self.processes.items():
            if process.is_alive():
                try:
                    proc = psutil.Process(process.pid)
                    process_info = {
                        'status': 'running',
                        'pid': process.pid,
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'create_time': proc.create_time(),
                    }
                    status['running_processes'] += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    process_info = {'status': 'unknown', 'pid': process.pid}
            else:
                process_info = {'status': 'dead', 'pid': process.pid, 'exit_code': process.exitcode}
                status['failed_processes'] += 1

            status['processes'][process_name] = process_info

        return status

    def print_system_status(self):
        """Print current system status."""
        status = self.get_system_status()

        print("\n" + "=" * 50)
        print("MCP SYSTEM STATUS")
        print("=" * 50)
        print(f"Total Processes: {status['total_processes']}")
        print(f"Running: {status['running_processes']}")
        print(f"Failed: {status['failed_processes']}")
        print("")

        for process_name, process_info in status['processes'].items():
            status_symbol = "✓" if process_info['status'] == 'running' else "✗"
            print(f"{status_symbol} {process_name:15} - {process_info['status']}")

            if process_info['status'] == 'running' and 'cpu_percent' in process_info:
                print(
                    f"    PID: {process_info['pid']}, "
                    f"CPU: {process_info['cpu_percent']:.1f}%, "
                    f"Memory: {process_info['memory_mb']:.1f}MB"
                )

        print("=" * 50 + "\n")


def main():
    """Main entry point for the orchestrator."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create and start orchestrator
    orchestrator = MCPSystemOrchestrator()

    try:
        orchestrator.start_system()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
