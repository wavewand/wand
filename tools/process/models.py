"""
Process Management Data Models

Defines data structures for process information, signals, and status tracking.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessStatus(str, Enum):
    """Process status enumeration"""

    RUNNING = "running"
    SLEEPING = "sleeping"
    DISK_SLEEP = "disk_sleep"
    STOPPED = "stopped"
    TRACING_STOP = "tracing_stop"
    ZOMBIE = "zombie"
    DEAD = "dead"
    WAKE_KILL = "wake_kill"
    WAKING = "waking"
    IDLE = "idle"
    LOCKED = "locked"
    WAITING = "waiting"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"


class ProcessSignal(str, Enum):
    """Common process signals"""

    SIGTERM = "SIGTERM"  # Terminate
    SIGKILL = "SIGKILL"  # Kill (force)
    SIGINT = "SIGINT"  # Interrupt
    SIGQUIT = "SIGQUIT"  # Quit
    SIGSTOP = "SIGSTOP"  # Stop
    SIGCONT = "SIGCONT"  # Continue
    SIGUSR1 = "SIGUSR1"  # User signal 1
    SIGUSR2 = "SIGUSR2"  # User signal 2
    SIGHUP = "SIGHUP"  # Hangup


@dataclass
class ProcessInfo:
    """Information about a system process"""

    pid: int
    ppid: Optional[int] = None
    name: str = ""
    cmdline: List[str] = field(default_factory=list)
    status: ProcessStatus = ProcessStatus.UNKNOWN
    username: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_info: Dict[str, int] = field(default_factory=dict)
    create_time: float = 0.0
    num_threads: int = 0
    open_files: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    cwd: str = ""
    exe: str = ""
    environ: Dict[str, str] = field(default_factory=dict)

    # Additional metadata
    backend_type: str = ""
    container_id: Optional[str] = None
    host: Optional[str] = None
    last_updated: float = field(default_factory=time.time)

    @property
    def command_line(self) -> str:
        """Get command line as string"""
        return ' '.join(self.cmdline) if self.cmdline else self.name

    @property
    def memory_mb(self) -> float:
        """Get memory usage in MB"""
        if 'rss' in self.memory_info:
            return self.memory_info['rss'] / (1024 * 1024)
        return 0.0

    @property
    def age_seconds(self) -> float:
        """Get process age in seconds"""
        if self.create_time > 0:
            return time.time() - self.create_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'pid': self.pid,
            'ppid': self.ppid,
            'name': self.name,
            'cmdline': self.cmdline,
            'status': self.status.value,
            'username': self.username,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_info': self.memory_info,
            'create_time': self.create_time,
            'num_threads': self.num_threads,
            'cwd': self.cwd,
            'exe': self.exe,
            'backend_type': self.backend_type,
            'container_id': self.container_id,
            'host': self.host,
            'last_updated': self.last_updated,
            'command_line': self.command_line,
            'memory_mb': self.memory_mb,
            'age_seconds': self.age_seconds,
        }


@dataclass
class ProcessTree:
    """Represents a process tree structure"""

    root: ProcessInfo
    children: List['ProcessTree'] = field(default_factory=list)

    def flatten(self) -> List[ProcessInfo]:
        """Flatten tree to list of processes"""
        processes = [self.root]
        for child in self.children:
            processes.extend(child.flatten())
        return processes

    def count_total(self) -> int:
        """Count total processes in tree"""
        return len(self.flatten())

    def find_by_pid(self, pid: int) -> Optional[ProcessInfo]:
        """Find process by PID in tree"""
        if self.root.pid == pid:
            return self.root
        for child in self.children:
            result = child.find_by_pid(pid)
            if result:
                return result
        return None


@dataclass
class ProcessFilter:
    """Filter criteria for process listing"""

    name_pattern: Optional[str] = None
    username: Optional[str] = None
    status: Optional[ProcessStatus] = None
    min_cpu_percent: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    min_memory_mb: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_age_seconds: Optional[float] = None
    max_age_seconds: Optional[float] = None
    backend_type: Optional[str] = None
    container_id: Optional[str] = None
    host: Optional[str] = None

    def matches(self, process: ProcessInfo) -> bool:
        """Check if process matches filter criteria"""
        import re

        if self.name_pattern and not re.search(self.name_pattern, process.name, re.IGNORECASE):
            return False

        if self.username and process.username != self.username:
            return False

        if self.status and process.status != self.status:
            return False

        if self.min_cpu_percent is not None and process.cpu_percent < self.min_cpu_percent:
            return False

        if self.max_cpu_percent is not None and process.cpu_percent > self.max_cpu_percent:
            return False

        if self.min_memory_mb is not None and process.memory_mb < self.min_memory_mb:
            return False

        if self.max_memory_mb is not None and process.memory_mb > self.max_memory_mb:
            return False

        if self.min_age_seconds is not None and process.age_seconds < self.min_age_seconds:
            return False

        if self.max_age_seconds is not None and process.age_seconds > self.max_age_seconds:
            return False

        if self.backend_type and process.backend_type != self.backend_type:
            return False

        if self.container_id and process.container_id != self.container_id:
            return False

        if self.host and process.host != self.host:
            return False

        return True


@dataclass
class ProcessOperation:
    """Represents a process operation result"""

    pid: int
    operation: str
    success: bool
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'pid': self.pid,
            'operation': self.operation,
            'success': self.success,
            'message': self.message,
            'timestamp': self.timestamp,
        }
