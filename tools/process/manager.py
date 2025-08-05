"""
Process Manager

Provides cross-platform process management capabilities with support for different
execution backends (native, SSH remote, Docker containers).
"""

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import psutil

from ..execution.base import ExecutionBackend, ExecutionConfig
from .models import ProcessFilter, ProcessInfo, ProcessOperation, ProcessSignal, ProcessStatus, ProcessTree

logger = logging.getLogger(__name__)


class ProcessBackend(ABC):
    """Abstract base class for process management backends"""

    @abstractmethod
    async def list_processes(self, filter_criteria: Optional[ProcessFilter] = None) -> List[ProcessInfo]:
        """List processes based on filter criteria"""
        pass

    @abstractmethod
    async def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed information about a specific process"""
        pass

    @abstractmethod
    async def kill_process(self, pid: int, signal_type: ProcessSignal = ProcessSignal.SIGTERM) -> ProcessOperation:
        """Send signal to a process"""
        pass

    @abstractmethod
    async def get_process_tree(self, pid: int) -> Optional[ProcessTree]:
        """Get process tree starting from given PID"""
        pass


class NativeProcessBackend(ProcessBackend):
    """Native process management using psutil"""

    def __init__(self):
        self.backend_type = "native"

    async def list_processes(self, filter_criteria: Optional[ProcessFilter] = None) -> List[ProcessInfo]:
        """List processes on local system"""
        processes = []

        try:
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'username', 'status', 'cmdline']):
                try:
                    process_info = await self._create_process_info(proc)

                    if filter_criteria is None or filter_criteria.matches(process_info):
                        processes.append(process_info)

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            logger.error(f"Error listing processes: {e}")

        return processes

    async def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed information about a specific process"""
        try:
            proc = psutil.Process(pid)
            return await self._create_process_info(proc, detailed=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
        except Exception as e:
            logger.error(f"Error getting process info for PID {pid}: {e}")
            return None

    async def kill_process(self, pid: int, signal_type: ProcessSignal = ProcessSignal.SIGTERM) -> ProcessOperation:
        """Send signal to a process"""
        try:
            proc = psutil.Process(pid)

            # Convert signal name to signal number
            sig_num = getattr(signal, signal_type.value, signal.SIGTERM)

            proc.send_signal(sig_num)

            return ProcessOperation(
                pid=pid,
                operation=f"kill_{signal_type.value}",
                success=True,
                message=f"Signal {signal_type.value} sent successfully",
            )

        except psutil.NoSuchProcess:
            return ProcessOperation(
                pid=pid, operation=f"kill_{signal_type.value}", success=False, message=f"Process {pid} not found"
            )
        except psutil.AccessDenied:
            return ProcessOperation(
                pid=pid, operation=f"kill_{signal_type.value}", success=False, message=f"Access denied to process {pid}"
            )
        except Exception as e:
            return ProcessOperation(
                pid=pid,
                operation=f"kill_{signal_type.value}",
                success=False,
                message=f"Error killing process: {str(e)}",
            )

    async def get_process_tree(self, pid: int) -> Optional[ProcessTree]:
        """Get process tree starting from given PID"""
        try:
            root_proc = psutil.Process(pid)
            root_info = await self._create_process_info(root_proc)

            tree = ProcessTree(root=root_info)
            await self._build_tree(tree, root_proc)

            return tree

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    async def _build_tree(self, tree: ProcessTree, proc: psutil.Process):
        """Recursively build process tree"""
        try:
            for child_proc in proc.children():
                try:
                    child_info = await self._create_process_info(child_proc)
                    child_tree = ProcessTree(root=child_info)
                    tree.children.append(child_tree)

                    # Recursively build subtree
                    await self._build_tree(child_tree, child_proc)

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.warning(f"Error building process tree: {e}")

    async def _create_process_info(self, proc: psutil.Process, detailed: bool = False) -> ProcessInfo:
        """Create ProcessInfo from psutil.Process"""
        try:
            # Basic info (always available)
            info = ProcessInfo(pid=proc.pid, backend_type=self.backend_type)

            # Safe attribute access with fallbacks
            try:
                info.ppid = proc.ppid()
            except BaseException:
                pass

            try:
                info.name = proc.name()
            except BaseException:
                pass

            try:
                info.cmdline = proc.cmdline()
            except BaseException:
                pass

            try:
                info.username = proc.username()
            except BaseException:
                pass

            try:
                status_map = {
                    psutil.STATUS_RUNNING: ProcessStatus.RUNNING,
                    psutil.STATUS_SLEEPING: ProcessStatus.SLEEPING,
                    psutil.STATUS_DISK_SLEEP: ProcessStatus.DISK_SLEEP,
                    psutil.STATUS_STOPPED: ProcessStatus.STOPPED,
                    psutil.STATUS_TRACING_STOP: ProcessStatus.TRACING_STOP,
                    psutil.STATUS_ZOMBIE: ProcessStatus.ZOMBIE,
                    psutil.STATUS_DEAD: ProcessStatus.DEAD,
                    psutil.STATUS_WAKE_KILL: ProcessStatus.WAKE_KILL,
                    psutil.STATUS_WAKING: ProcessStatus.WAKING,
                    psutil.STATUS_IDLE: ProcessStatus.IDLE,
                    psutil.STATUS_LOCKED: ProcessStatus.LOCKED,
                    psutil.STATUS_WAITING: ProcessStatus.WAITING,
                    psutil.STATUS_SUSPENDED: ProcessStatus.SUSPENDED,
                }
                info.status = status_map.get(proc.status(), ProcessStatus.UNKNOWN)
            except BaseException:
                pass

            try:
                info.create_time = proc.create_time()
            except BaseException:
                pass

            # Detailed info (if requested and available)
            if detailed:
                try:
                    info.cpu_percent = proc.cpu_percent()
                except BaseException:
                    pass

                try:
                    info.memory_percent = proc.memory_percent()
                except BaseException:
                    pass

                try:
                    memory_info = proc.memory_info()
                    info.memory_info = {'rss': memory_info.rss, 'vms': memory_info.vms}
                except BaseException:
                    pass

                try:
                    info.num_threads = proc.num_threads()
                except BaseException:
                    pass

                try:
                    info.cwd = proc.cwd()
                except BaseException:
                    pass

                try:
                    info.exe = proc.exe()
                except BaseException:
                    pass

                try:
                    info.environ = proc.environ()
                except BaseException:
                    pass

                try:
                    info.open_files = [{'path': f.path, 'fd': f.fd, 'mode': f.mode} for f in proc.open_files()]
                except BaseException:
                    pass

                try:
                    info.connections = [
                        {
                            'fd': conn.fd,
                            'family': conn.family.name if hasattr(conn.family, 'name') else str(conn.family),
                            'type': conn.type.name if hasattr(conn.type, 'name') else str(conn.type),
                            'laddr': conn.laddr,
                            'raddr': conn.raddr,
                            'status': conn.status,
                        }
                        for conn in proc.connections()
                    ]
                except BaseException:
                    pass

            return info

        except Exception as e:
            logger.error(f"Error creating process info: {e}")
            # Return minimal info
            return ProcessInfo(pid=proc.pid, backend_type=self.backend_type)


class RemoteProcessBackend(ProcessBackend):
    """Remote process management via execution backend"""

    def __init__(self, execution_backend: ExecutionBackend, host: str = "remote"):
        self.execution_backend = execution_backend
        self.host = host
        self.backend_type = f"remote_{host}"

    async def list_processes(self, filter_criteria: Optional[ProcessFilter] = None) -> List[ProcessInfo]:
        """List processes on remote system"""
        try:
            # Use ps command to get process list
            ps_config = ExecutionConfig(
                command="ps -eo pid,ppid,user,comm,state,pcpu,pmem,lstart,cmd --no-headers", timeout=30
            )

            result = await self.execution_backend.execute(ps_config)

            if not result.success:
                logger.error(f"Failed to list remote processes: {result.stderr}")
                return []

            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    process_info = self._parse_ps_output(line)
                    if process_info and (filter_criteria is None or filter_criteria.matches(process_info)):
                        processes.append(process_info)

            return processes

        except Exception as e:
            logger.error(f"Error listing remote processes: {e}")
            return []

    async def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed information about a specific remote process"""
        try:
            # Get basic process info
            ps_config = ExecutionConfig(
                command=f"ps -p {pid} -o pid,ppid,user,comm,state,pcpu,pmem,lstart,cmd --no-headers", timeout=30
            )

            result = await self.execution_backend.execute(ps_config)

            if not result.success:
                return None

            if not result.stdout.strip():
                return None

            process_info = self._parse_ps_output(result.stdout.strip())

            # Get additional details
            if process_info:
                await self._enrich_process_info(process_info)

            return process_info

        except Exception as e:
            logger.error(f"Error getting remote process info for PID {pid}: {e}")
            return None

    async def kill_process(self, pid: int, signal_type: ProcessSignal = ProcessSignal.SIGTERM) -> ProcessOperation:
        """Send signal to a remote process"""
        try:
            kill_config = ExecutionConfig(command=f"kill -{signal_type.value} {pid}", timeout=10)

            result = await self.execution_backend.execute(kill_config)

            if result.success:
                return ProcessOperation(
                    pid=pid,
                    operation=f"kill_{signal_type.value}",
                    success=True,
                    message=f"Signal {signal_type.value} sent successfully",
                )
            else:
                return ProcessOperation(
                    pid=pid,
                    operation=f"kill_{signal_type.value}",
                    success=False,
                    message=result.stderr or "Unknown error",
                )

        except Exception as e:
            return ProcessOperation(
                pid=pid,
                operation=f"kill_{signal_type.value}",
                success=False,
                message=f"Error killing remote process: {str(e)}",
            )

    async def get_process_tree(self, pid: int) -> Optional[ProcessTree]:
        """Get process tree for remote process"""
        try:
            # Use pstree or ps to get process tree
            tree_config = ExecutionConfig(
                command=f"pstree -p {pid} 2>/dev/null || ps --forest -o pid,ppid,cmd -g {pid}", timeout=30
            )

            result = await self.execution_backend.execute(tree_config)

            if not result.success:
                return None

            # For now, just return single process (can be enhanced to parse tree structure)
            root_info = await self.get_process_info(pid)
            if root_info:
                return ProcessTree(root=root_info)

            return None

        except Exception as e:
            logger.error(f"Error getting remote process tree for PID {pid}: {e}")
            return None

    def _parse_ps_output(self, line: str) -> Optional[ProcessInfo]:
        """Parse ps command output line"""
        try:
            parts = line.split(None, 8)  # Split into max 9 parts

            if len(parts) < 8:
                return None

            pid = int(parts[0])
            ppid = int(parts[1]) if parts[1] != '0' else None
            username = parts[2]
            name = parts[3]
            state = parts[4]
            cpu_percent = float(parts[5])
            memory_percent = float(parts[6])
            start_time = parts[7]
            cmdline = parts[8] if len(parts) > 8 else name

            # Map process state
            state_map = {
                'R': ProcessStatus.RUNNING,
                'S': ProcessStatus.SLEEPING,
                'D': ProcessStatus.DISK_SLEEP,
                'T': ProcessStatus.STOPPED,
                't': ProcessStatus.TRACING_STOP,
                'Z': ProcessStatus.ZOMBIE,
                'X': ProcessStatus.DEAD,
                'I': ProcessStatus.IDLE,
                'W': ProcessStatus.WAKING,
            }

            status = state_map.get(state[0], ProcessStatus.UNKNOWN)

            return ProcessInfo(
                pid=pid,
                ppid=ppid,
                name=name,
                cmdline=cmdline.split(),
                status=status,
                username=username,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                backend_type=self.backend_type,
                host=self.host,
            )

        except Exception as e:
            logger.warning(f"Error parsing ps output '{line}': {e}")
            return None

    async def _enrich_process_info(self, process_info: ProcessInfo):
        """Add additional details to process info"""
        try:
            # Get working directory
            pwd_config = ExecutionConfig(
                command=f"pwdx {process_info.pid} 2>/dev/null | cut -d: -f2 | xargs", timeout=10
            )

            result = await self.execution_backend.execute(pwd_config)
            if result.success and result.stdout.strip():
                process_info.cwd = result.stdout.strip()

            # Get executable path
            exe_config = ExecutionConfig(command=f"readlink -f /proc/{process_info.pid}/exe 2>/dev/null", timeout=10)

            result = await self.execution_backend.execute(exe_config)
            if result.success and result.stdout.strip():
                process_info.exe = result.stdout.strip()

        except Exception as e:
            logger.debug(f"Error enriching process info: {e}")


class ProcessManager:
    """Main process manager that coordinates different backends"""

    def __init__(self):
        self.backends: Dict[str, ProcessBackend] = {}
        self.default_backend = "native"

        # Initialize native backend
        self.backends["native"] = NativeProcessBackend()

    def add_backend(self, name: str, backend: ProcessBackend):
        """Add a process management backend"""
        self.backends[name] = backend
        logger.info(f"Added process backend: {name}")

    def add_execution_backend(self, name: str, execution_backend: ExecutionBackend, host: str = "remote"):
        """Add remote process backend using execution backend"""
        remote_backend = RemoteProcessBackend(execution_backend, host)
        self.add_backend(name, remote_backend)

    async def list_processes(
        self, backend: Optional[str] = None, filter_criteria: Optional[ProcessFilter] = None
    ) -> List[ProcessInfo]:
        """List processes from specified backend or all backends"""
        if backend:
            if backend not in self.backends:
                raise ValueError(f"Unknown backend: {backend}")
            return await self.backends[backend].list_processes(filter_criteria)

        # List from all backends
        all_processes = []
        for backend_name, backend_instance in self.backends.items():
            try:
                processes = await backend_instance.list_processes(filter_criteria)
                all_processes.extend(processes)
            except Exception as e:
                logger.error(f"Error listing processes from backend {backend_name}: {e}")

        return all_processes

    async def get_process_info(self, pid: int, backend: Optional[str] = None) -> Optional[ProcessInfo]:
        """Get process information from specified backend"""
        backend_name = backend or self.default_backend

        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")

        return await self.backends[backend_name].get_process_info(pid)

    async def kill_process(
        self, pid: int, signal_type: ProcessSignal = ProcessSignal.SIGTERM, backend: Optional[str] = None
    ) -> ProcessOperation:
        """Kill process using specified backend"""
        backend_name = backend or self.default_backend

        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")

        return await self.backends[backend_name].kill_process(pid, signal_type)

    async def get_process_tree(self, pid: int, backend: Optional[str] = None) -> Optional[ProcessTree]:
        """Get process tree from specified backend"""
        backend_name = backend or self.default_backend

        if backend_name not in self.backends:
            raise ValueError(f"Unknown backend: {backend_name}")

        return await self.backends[backend_name].get_process_tree(pid)

    def list_backends(self) -> List[str]:
        """List available process backends"""
        return list(self.backends.keys())

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all backends"""
        health = {}
        for name, backend in self.backends.items():
            try:
                # Simple health check - try to list processes
                processes = await backend.list_processes(ProcessFilter())
                health[name] = len(processes) >= 0  # Should return list (empty or not)
            except Exception as e:
                logger.error(f"Health check failed for backend {name}: {e}")
                health[name] = False

        return health
