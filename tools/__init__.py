# MCP-Python Tools Package
# Comprehensive toolset for system command execution with multiple deployment modes

# Import available modules
from .execution import *
from .host_agent import *
from .process import *

__version__ = "1.0.0"
__author__ = "MCP-Python Development Team"

# Tool categories
SYSTEM_TOOLS = [
    "execute_command",
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "list_processes",
    "kill_process",
    "get_process_info",
    "get_process_tree",
    "monitor_processes",
    "get_system_info",
    "get_environment_vars",
    "check_command_exists",
]

SECURITY_TOOLS = ["validate_command", "check_permissions", "audit_log", "sandbox_execute"]

PLATFORM_TOOLS = ["mac_specific_tools", "linux_specific_tools", "unix_common_tools"]

ALL_TOOLS = SYSTEM_TOOLS + SECURITY_TOOLS + PLATFORM_TOOLS
