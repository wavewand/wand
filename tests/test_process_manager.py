"""
Tests for Process Management Tools
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest

from tools.execution.base import ExecutionBackend, ExecutionConfig, ExecutionResult, ExecutionStatus
from tools.process.manager import NativeProcessBackend, ProcessManager, RemoteProcessBackend
from tools.process.models import ProcessFilter, ProcessInfo, ProcessOperation, ProcessSignal, ProcessStatus, ProcessTree
from tools.process.monitor import AlertLevel, MonitoringRule, ProcessAlert, ProcessMonitor, create_process_monitor


class TestProcessModels:
    """Test process data models"""

    def test_process_info_creation(self):
        """Test ProcessInfo creation and properties"""
        process_info = ProcessInfo(
            pid=1234,
            ppid=1,
            name="test_process",
            cmdline=["test_process", "--arg1", "value1"],
            status=ProcessStatus.RUNNING,
            username="testuser",
            cpu_percent=15.5,
            memory_percent=8.2,
            memory_info={"rss": 1048576, "vms": 2097152},  # 1MB RSS, 2MB VMS
            create_time=time.time() - 3600,  # 1 hour ago
            backend_type="native",
        )

        assert process_info.pid == 1234
        assert process_info.command_line == "test_process --arg1 value1"
        assert process_info.memory_mb == 1.0  # 1MB
        assert process_info.age_seconds > 3590  # ~1 hour

        # Test to_dict conversion
        data = process_info.to_dict()
        assert data["pid"] == 1234
        assert data["status"] == "running"
        assert data["command_line"] == "test_process --arg1 value1"

    def test_process_filter_matching(self):
        """Test ProcessFilter matching logic"""
        process = ProcessInfo(
            pid=1234,
            name="python",
            username="testuser",
            status=ProcessStatus.RUNNING,
            cpu_percent=25.0,
            memory_info={"rss": 2097152},  # 2MB
            create_time=time.time() - 1800,  # 30 minutes ago
            backend_type="native",
        )

        # Test name pattern matching
        filter1 = ProcessFilter(name_pattern="python")
        assert filter1.matches(process) is True

        filter2 = ProcessFilter(name_pattern="java")
        assert filter2.matches(process) is False

        # Test CPU threshold
        filter3 = ProcessFilter(min_cpu_percent=20.0, max_cpu_percent=30.0)
        assert filter3.matches(process) is True

        filter4 = ProcessFilter(min_cpu_percent=50.0)
        assert filter4.matches(process) is False

        # Test memory threshold
        filter5 = ProcessFilter(min_memory_mb=1.0, max_memory_mb=3.0)
        assert filter5.matches(process) is True

    def test_process_tree_operations(self):
        """Test ProcessTree operations"""
        root = ProcessInfo(pid=1, name="init")
        child1 = ProcessInfo(pid=100, ppid=1, name="child1")
        child2 = ProcessInfo(pid=200, ppid=1, name="child2")
        grandchild = ProcessInfo(pid=300, ppid=100, name="grandchild")

        tree = ProcessTree(root=root)
        child1_tree = ProcessTree(root=child1)
        child2_tree = ProcessTree(root=child2)
        grandchild_tree = ProcessTree(root=grandchild)

        child1_tree.children.append(grandchild_tree)
        tree.children.extend([child1_tree, child2_tree])

        # Test flattening
        flat = tree.flatten()
        assert len(flat) == 4
        assert root in flat
        assert grandchild in flat

        # Test counting
        assert tree.count_total() == 4

        # Test finding by PID
        found = tree.find_by_pid(300)
        assert found == grandchild

        not_found = tree.find_by_pid(999)
        assert not_found is None


class TestNativeProcessBackend:
    """Test native process backend using psutil"""

    @pytest.fixture
    def backend(self):
        """Create native process backend"""
        return NativeProcessBackend()

    @patch('tools.process.manager.psutil.process_iter')
    @pytest.mark.asyncio
    async def test_list_processes(self, mock_process_iter, backend):
        """Test listing processes"""
        # Mock processes
        mock_proc1 = Mock()
        mock_proc1.pid = 1234
        mock_proc1.ppid.return_value = 1
        mock_proc1.name.return_value = "test_process"
        mock_proc1.username.return_value = "testuser"
        mock_proc1.status.return_value = psutil.STATUS_RUNNING
        mock_proc1.cmdline.return_value = ["test_process"]
        mock_proc1.create_time.return_value = time.time()

        mock_process_iter.return_value = [mock_proc1]

        processes = await backend.list_processes()

        assert len(processes) == 1
        assert processes[0].pid == 1234
        assert processes[0].name == "test_process"
        assert processes[0].backend_type == "native"

    @patch('tools.process.manager.psutil.Process')
    @pytest.mark.asyncio
    async def test_get_process_info(self, mock_process_class, backend):
        """Test getting detailed process info"""
        # Mock process
        mock_proc = Mock()
        mock_proc.pid = 1234
        mock_proc.ppid.return_value = 1
        mock_proc.name.return_value = "test_process"
        mock_proc.username.return_value = "testuser"
        mock_proc.status.return_value = psutil.STATUS_RUNNING
        mock_proc.cmdline.return_value = ["test_process", "--arg"]
        mock_proc.create_time.return_value = time.time()
        mock_proc.cpu_percent.return_value = 15.5
        mock_proc.memory_percent.return_value = 8.2
        mock_proc.memory_info.return_value = Mock(rss=1048576, vms=2097152)
        mock_proc.num_threads.return_value = 4
        mock_proc.cwd.return_value = "/home/test"
        mock_proc.exe.return_value = "/usr/bin/test_process"

        mock_process_class.return_value = mock_proc

        process_info = await backend.get_process_info(1234)

        assert process_info is not None
        assert process_info.pid == 1234
        assert process_info.cpu_percent == 15.5
        assert process_info.memory_mb == 1.0
        assert process_info.num_threads == 4
        assert process_info.cwd == "/home/test"

    @patch('tools.process.manager.psutil.Process')
    @pytest.mark.asyncio
    async def test_kill_process_success(self, mock_process_class, backend):
        """Test successful process termination"""
        mock_proc = Mock()
        mock_proc.send_signal = Mock()
        mock_process_class.return_value = mock_proc

        result = await backend.kill_process(1234, ProcessSignal.SIGTERM)

        assert result.success is True
        assert result.pid == 1234
        assert "sent successfully" in result.message
        mock_proc.send_signal.assert_called_once()

    @patch('tools.process.manager.psutil.Process')
    @pytest.mark.asyncio
    async def test_kill_process_not_found(self, mock_process_class, backend):
        """Test killing non-existent process"""
        mock_process_class.side_effect = psutil.NoSuchProcess(1234)

        result = await backend.kill_process(1234, ProcessSignal.SIGTERM)

        assert result.success is False
        assert "not found" in result.message

    @patch('tools.process.manager.psutil.Process')
    @pytest.mark.asyncio
    async def test_get_process_tree(self, mock_process_class, backend):
        """Test getting process tree"""
        # Mock root process
        mock_root = Mock()
        mock_root.pid = 1234
        mock_root.name.return_value = "parent"
        mock_root.cmdline.return_value = ["parent"]

        # Mock child process
        mock_child = Mock()
        mock_child.pid = 5678
        mock_child.name.return_value = "child"
        mock_child.cmdline.return_value = ["child"]
        mock_child.children.return_value = []

        mock_root.children.return_value = [mock_child]
        mock_process_class.return_value = mock_root

        tree = await backend.get_process_tree(1234)

        assert tree is not None
        assert tree.root.pid == 1234
        assert len(tree.children) == 1
        assert tree.children[0].root.pid == 5678


class TestRemoteProcessBackend:
    """Test remote process backend"""

    @pytest.fixture
    def mock_execution_backend(self):
        """Create mock execution backend"""
        backend = Mock(spec=ExecutionBackend)
        return backend

    @pytest.fixture
    def backend(self, mock_execution_backend):
        """Create remote process backend"""
        return RemoteProcessBackend(mock_execution_backend, "test-host")

    @pytest.mark.asyncio
    async def test_list_processes(self, backend, mock_execution_backend):
        """Test listing remote processes"""
        # Mock execution result with ps output
        ps_output = (
            "1234 1 root systemd R 0.1 0.5 Jan01 /sbin/init\n5678 1234 user python S 2.5 1.2 10:30 python script.py"
        )

        mock_result = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=ps_output, stderr="", exit_code=0)

        mock_execution_backend.execute.return_value = mock_result

        processes = await backend.list_processes()

        assert len(processes) == 2
        assert processes[0].pid == 1234
        assert processes[0].name == "systemd"
        assert processes[0].username == "root"
        assert processes[1].pid == 5678
        assert processes[1].name == "python"
        assert processes[1].cpu_percent == 2.5

    @pytest.mark.asyncio
    async def test_get_process_info(self, backend, mock_execution_backend):
        """Test getting remote process info"""
        ps_output = "1234 1 testuser python S 1.5 2.0 10:30 python test.py"

        mock_result = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=ps_output, stderr="", exit_code=0)

        mock_execution_backend.execute.return_value = mock_result

        process_info = await backend.get_process_info(1234)

        assert process_info is not None
        assert process_info.pid == 1234
        assert process_info.name == "python"
        assert process_info.username == "testuser"
        assert process_info.backend_type == "remote_test-host"

    @pytest.mark.asyncio
    async def test_kill_process(self, backend, mock_execution_backend):
        """Test killing remote process"""
        mock_result = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="", stderr="", exit_code=0)

        mock_execution_backend.execute.return_value = mock_result

        result = await backend.kill_process(1234, ProcessSignal.SIGTERM)

        assert result.success is True
        assert result.pid == 1234

        # Verify correct kill command was executed
        call_args = mock_execution_backend.execute.call_args[0][0]
        assert "kill -SIGTERM 1234" in call_args.command


class TestProcessManager:
    """Test process manager coordination"""

    @pytest.fixture
    def manager(self):
        """Create process manager"""
        return ProcessManager()

    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert "native" in manager.backends
        assert manager.default_backend == "native"
        assert len(manager.list_backends()) == 1

    def test_add_backend(self, manager):
        """Test adding custom backend"""
        mock_backend = Mock(spec=RemoteProcessBackend)
        manager.add_backend("test-remote", mock_backend)

        assert "test-remote" in manager.backends
        assert len(manager.list_backends()) == 2

    def test_add_execution_backend(self, manager):
        """Test adding execution backend"""
        mock_execution_backend = Mock(spec=ExecutionBackend)
        manager.add_execution_backend("ssh-remote", mock_execution_backend, "remote-host")

        assert "ssh-remote" in manager.backends
        backend = manager.backends["ssh-remote"]
        assert isinstance(backend, RemoteProcessBackend)
        assert backend.host == "remote-host"

    @pytest.mark.asyncio
    async def test_list_processes_single_backend(self, manager):
        """Test listing processes from single backend"""
        mock_backend = AsyncMock()
        mock_processes = [ProcessInfo(pid=1234, name="test")]
        mock_backend.list_processes.return_value = mock_processes

        manager.add_backend("test", mock_backend)

        processes = await manager.list_processes("test")

        assert processes == mock_processes
        mock_backend.list_processes.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_processes_all_backends(self, manager):
        """Test listing processes from all backends"""
        mock_backend1 = AsyncMock()
        mock_backend1.list_processes.return_value = [ProcessInfo(pid=1234, name="test1")]

        mock_backend2 = AsyncMock()
        mock_backend2.list_processes.return_value = [ProcessInfo(pid=5678, name="test2")]

        manager.add_backend("test1", mock_backend1)
        manager.add_backend("test2", mock_backend2)

        processes = await manager.list_processes()

        assert len(processes) >= 2  # Native backend + 2 test backends
        pids = [p.pid for p in processes]
        assert 1234 in pids
        assert 5678 in pids

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        """Test health check of all backends"""
        mock_backend = AsyncMock()
        mock_backend.list_processes.return_value = [ProcessInfo(pid=1234, name="test")]

        manager.add_backend("test", mock_backend)

        health = await manager.health_check()

        assert "native" in health
        assert "test" in health
        assert health["test"] is True


class TestProcessMonitor:
    """Test process monitoring system"""

    @pytest.fixture
    def process_manager(self):
        """Create mock process manager"""
        manager = Mock(spec=ProcessManager)
        manager.list_backends.return_value = ["native"]
        return manager

    @pytest.fixture
    def monitor(self, process_manager):
        """Create process monitor"""
        return ProcessMonitor(process_manager)

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.is_running is False
        assert len(monitor.rules) == 0
        assert len(monitor.alert_handlers) == 0

    def test_add_rule(self, monitor):
        """Test adding monitoring rule"""
        rule = MonitoringRule(
            name="test_rule",
            description="Test rule",
            filter_criteria=ProcessFilter(name_pattern="python"),
            max_cpu_percent=80.0,
        )

        monitor.add_rule(rule)

        assert "test_rule" in monitor.rules
        assert monitor.rules["test_rule"] == rule

    def test_remove_rule(self, monitor):
        """Test removing monitoring rule"""
        rule = MonitoringRule(name="test_rule", description="Test rule", filter_criteria=ProcessFilter())

        monitor.add_rule(rule)
        assert "test_rule" in monitor.rules

        removed = monitor.remove_rule("test_rule")
        assert removed is True
        assert "test_rule" not in monitor.rules

        # Test removing non-existent rule
        removed = monitor.remove_rule("nonexistent")
        assert removed is False

    def test_add_alert_handler(self, monitor):
        """Test adding alert handler"""

        def test_handler(alert):
            pass

        monitor.add_alert_handler(test_handler)

        assert test_handler in monitor.alert_handlers
        assert len(monitor.alert_handlers) == 1

    def test_monitoring_rule_check_cpu_threshold(self):
        """Test monitoring rule CPU threshold checking"""
        rule = MonitoringRule(
            name="cpu_rule", description="CPU threshold rule", filter_criteria=ProcessFilter(), max_cpu_percent=50.0
        )

        # Process exceeding threshold
        high_cpu_process = ProcessInfo(pid=1234, name="high_cpu", cpu_percent=75.0)

        # Process within threshold
        low_cpu_process = ProcessInfo(pid=5678, name="low_cpu", cpu_percent=25.0)

        alerts = rule.check_processes([high_cpu_process, low_cpu_process], [])

        assert len(alerts) == 1
        assert alerts[0].process_info.pid == 1234
        assert alerts[0].alert_type == "high_cpu"
        assert alerts[0].level == AlertLevel.WARNING

    def test_monitoring_rule_check_process_count(self):
        """Test monitoring rule process count checking"""
        rule = MonitoringRule(
            name="count_rule",
            description="Process count rule",
            filter_criteria=ProcessFilter(name_pattern="python"),
            min_processes=2,
            max_processes=5,
        )

        # Only one python process (below minimum)
        processes = [ProcessInfo(pid=1234, name="python")]

        alerts = rule.check_processes(processes, [])

        assert len(alerts) == 1
        assert alerts[0].alert_type == "process_count_low"
        assert alerts[0].actual_value == 1
        assert alerts[0].threshold_value == 2

    def test_monitoring_rule_lifecycle_alerts(self):
        """Test process lifecycle alerts"""
        rule = MonitoringRule(
            name="lifecycle_rule",
            description="Process lifecycle rule",
            filter_criteria=ProcessFilter(name_pattern="important"),
            alert_on_new_process=True,
            alert_on_process_exit=True,
        )

        # Previous state: one process
        previous = [ProcessInfo(pid=1234, name="important_service")]

        # Current state: different process (1234 exited, 5678 started)
        current = [ProcessInfo(pid=5678, name="important_service")]

        alerts = rule.check_processes(current, previous)

        assert len(alerts) == 2
        alert_types = [a.alert_type for a in alerts]
        assert "process_started" in alert_types
        assert "process_exited" in alert_types

    @pytest.mark.asyncio
    async def test_alert_handling(self, monitor):
        """Test alert handling"""
        handled_alerts = []

        def test_handler(alert):
            handled_alerts.append(alert)

        monitor.add_alert_handler(test_handler)

        alert = ProcessAlert(
            level=AlertLevel.WARNING, message="Test alert", process_info=ProcessInfo(pid=1234, name="test")
        )

        await monitor._handle_alert(alert)

        assert len(handled_alerts) == 1
        assert handled_alerts[0] == alert
        assert len(monitor.alert_history) == 1

    def test_get_recent_alerts(self, monitor):
        """Test getting recent alerts"""
        # Add some alerts to history
        alerts = [
            ProcessAlert(AlertLevel.INFO, "Info alert", ProcessInfo(pid=1, name="test")),
            ProcessAlert(AlertLevel.WARNING, "Warning alert", ProcessInfo(pid=2, name="test")),
            ProcessAlert(AlertLevel.CRITICAL, "Critical alert", ProcessInfo(pid=3, name="test")),
        ]

        monitor.alert_history.extend(alerts)

        # Test getting all recent alerts
        recent = monitor.get_recent_alerts(10)
        assert len(recent) == 3

        # Test filtering by level
        warnings = monitor.get_recent_alerts(10, AlertLevel.WARNING)
        assert len(warnings) == 1
        assert warnings[0].level == AlertLevel.WARNING

    def test_get_stats(self, monitor):
        """Test getting monitoring statistics"""
        stats = monitor.get_stats()

        assert "is_running" in stats
        assert "active_rules" in stats
        assert "total_rules" in stats
        assert "checks_performed" in stats
        assert "alerts_generated" in stats

        assert stats["is_running"] is False
        assert stats["active_rules"] == 0
        assert stats["total_rules"] == 0

    def test_create_default_rules(self, monitor):
        """Test creating default monitoring rules"""
        monitor.create_default_rules()

        assert len(monitor.rules) > 0
        assert "high_cpu_usage" in monitor.rules
        assert "high_memory_usage" in monitor.rules
        assert "zombie_processes" in monitor.rules
        assert "important_services" in monitor.rules

    def test_create_process_monitor_factory(self):
        """Test process monitor factory function"""
        mock_manager = Mock(spec=ProcessManager)

        monitor = create_process_monitor(mock_manager, with_defaults=True)

        assert isinstance(monitor, ProcessMonitor)
        assert len(monitor.rules) > 0  # Should have default rules
        assert len(monitor.alert_handlers) > 0  # Should have default handler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
