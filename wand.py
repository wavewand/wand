#!/usr/bin/env python3
"""
MCP Multi-Agent System - Main Entry Point

New architecture with separated concerns:
- Transport layer: MCP HTTP/stdio and gRPC
- Orchestrator layer: Agent management and task distribution
- Agent layer: Individual agents with all MCP tools
- API layer: REST endpoints for web dashboard
- Core layer: Task management and distributed system logic
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import uvicorn
from mcp.server import FastMCP
from mcp.types import ToolAnnotations

# Setup enhanced logging system
from observability.enhanced_logging import (
    LogCategory,
    LogLevel,
    SystemLogConfig,
    get_enhanced_logger,
    get_mcp_logger,
    initialize_enhanced_logging,
    log_mcp_call,
    log_tool_execution,
)
from observability.log_management import LogManagementConfig, LogManagementSystem
from orchestrator.agent_orchestrator import AgentOrchestrator
from transport.mcp_http import MCPHttpTransport
from transport.mcp_sse import MCPSSETransport

# Initialize enhanced logging with system-level directories
stdio_mode = len(sys.argv) > 1 and sys.argv[1] == "stdio"

# Configure enhanced logging
log_config = SystemLogConfig(
    level=LogLevel.INFO,
    enable_stdio_safety=True,
    use_system_logs=True,
    use_single_log_file=True,  # Use single log file for now
    log_mcp_requests=True,
    log_tool_calls=True,
    log_tool_inputs=True,
    log_tool_outputs=True,
    log_performance_metrics=True,
    log_security_events=True,
    log_audit_trail=True,
)

# Initialize the enhanced logging system
enhanced_logging_system = initialize_enhanced_logging(log_config)

# Setup log management
log_mgmt_config = LogManagementConfig(
    enabled=True,
    check_interval_minutes=30,
    compress_after_hours=24,
    enable_health_monitoring=True,
    max_log_file_size_mb=100,
    max_total_log_size_gb=5,
)

log_management_system = LogManagementSystem(log_mgmt_config, enhanced_logging_system.log_directory)

# Get loggers
logger = get_enhanced_logger(LogCategory.SYSTEM)
mcp_logger = get_mcp_logger()

# Import configuration - use simple config for now


class SimpleConfig:
    def __init__(self):
        self.server = SimpleServerConfig()


class SimpleServerConfig:
    def __init__(self):
        self.version = "1.0.0"
        self.mcp_http_port = 8001


config = SimpleConfig()

# Import transport layer

# Import orchestrator

# Import MCP server for stdio transport

# Import HTTP transport for OpenCode


def process_kwargs_dual_mode(**kwargs) -> Dict[str, Any]:
    """
    Process kwargs to support both direct dict parameters and JSON string via 'kwargs' key.

    This enables dual-mode parameter support for Claude MCP compatibility:
    - Mode 1: Direct kwargs (Python/dict style): func(param1="value1", param2="value2")
    - Mode 2: JSON string (Claude MCP style): func(kwargs='{"param1": "value1", "param2": "value2"}')
    - Mode 3: Key=value pairs (fallback): func(kwargs="param1=value1,param2=value2")

    Args:
        **kwargs: Raw keyword arguments that may include a 'kwargs' key with JSON string

    Returns:
        Dict[str, Any]: Processed parameters ready for use
    """
    import json

    final_kwargs = {}

    # Check if 'kwargs' key contains a JSON string (Claude MCP mode)
    if 'kwargs' in kwargs:
        kwargs_value = kwargs.pop('kwargs')  # Remove 'kwargs' key

        if isinstance(kwargs_value, str):
            try:
                # Try to parse as JSON
                if kwargs_value and kwargs_value != "{}":
                    parsed_kwargs = json.loads(kwargs_value)
                    final_kwargs.update(parsed_kwargs)
            except json.JSONDecodeError:
                # If not JSON, try parsing as key=value pairs
                try:
                    for pair in kwargs_value.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            final_kwargs[key.strip()] = value.strip()
                except BaseException:
                    pass
        elif isinstance(kwargs_value, dict):
            # If already a dict, use it directly
            final_kwargs.update(kwargs_value)

    # Add any remaining direct kwargs (Python/dict mode)
    final_kwargs.update(kwargs)

    return final_kwargs


async def create_orchestrator_stdio_server(orchestrator: AgentOrchestrator) -> FastMCP:
    """Create Wand MCP server - a magical toolkit for file operations, system commands, project management, and integrations"""
    mcp_server = FastMCP("wand")

    # Register specific tools with proper parameters instead of dynamic registration
    @mcp_server.tool(
        name="run",
        description="Execute shell command (e.g., ls, python script.py, mkdir, cat file.txt)",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def run(command: str, **kwargs) -> Dict[str, Any]:
        """Execute shell command (e.g., ls, python script.py, mkdir, cat file.txt)

        Args:
            command: Shell command to execute (e.g., 'ls -la', 'python hello.py', 'mkdir mydir')
            **kwargs: Additional parameters (supports both dict and JSON string via 'kwargs' key)

        Returns:
            Dict containing command output, exit code, and execution status

        Raises:
            Error if command fails to execute or times out
        """
        final_kwargs = process_kwargs_dual_mode(**kwargs)
        result = await orchestrator.execute_tool("execute_command", {"command": command, **final_kwargs})
        return result if isinstance(result, dict) else {"output": str(result), "status": "success"}

    @mcp_server.tool(
        name="read", description="Read contents of a file", annotations=ToolAnnotations(category="file_operations")
    )
    async def read(file_path: str, **kwargs) -> Dict[str, Any]:
        """Read contents of a file

        Args:
            file_path: Path to the file to read (e.g., 'hello.py', '/path/to/file.txt')
            **kwargs: Additional parameters (supports both dict and JSON string via 'kwargs' key)

        Returns:
            Dict containing file contents and metadata

        Raises:
            Error if file not found, permission denied, or file too large
        """
        final_kwargs = process_kwargs_dual_mode(**kwargs)
        result = await orchestrator.execute_tool("read_file", {"file_path": file_path, **final_kwargs})
        return (
            result
            if isinstance(result, dict)
            else {"content": str(result), "file_path": file_path, "status": "success"}
        )

    @mcp_server.tool(
        name="write",
        description="Write content to a file (creates new file or overwrites existing)",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def write(file_path: str, content: str, **kwargs) -> Dict[str, Any]:
        """Write content to a file (creates new file or overwrites existing)

        Args:
            file_path: Path where to write the file (e.g., 'hello.py', '/path/to/output.txt')
            content: Content to write to the file (e.g., 'print("Hello World!")')
            **kwargs: Additional parameters (supports both dict and JSON string via 'kwargs' key)

        Returns:
            Dict containing write status, file size, and path confirmation

        Raises:
            Error if permission denied, disk full, or invalid file path
        """
        final_kwargs = process_kwargs_dual_mode(**kwargs)
        result = await orchestrator.execute_tool(
            "write_file", {"file_path": file_path, "content": content, **final_kwargs}
        )
        return (
            result
            if isinstance(result, dict)
            else {"status": "success", "file_path": file_path, "bytes_written": len(content)}
        )

    @mcp_server.tool(
        name="list",
        description="List files and folders in a directory",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def list(**kwargs) -> Dict[str, Any]:
        """List files and folders in a directory

        Args:
            **kwargs: Directory listing options
                - directory: Directory path to list (defaults to current directory '.')
                - show_hidden: Include hidden files (defaults to False)
                - recursive: List recursively (defaults to False)
                - filter_pattern: Filter files by pattern

        Examples:
            list()  # List current directory
            list(directory="/home/user/documents")
            list(directory="/var/log", show_hidden=True, filter_pattern="*.log")
        """
        directory = kwargs.get("directory", ".")
        result = await orchestrator.execute_tool("list_directory", {"directory": directory, **kwargs})
        return (
            result if isinstance(result, dict) else {"files": str(result), "directory": directory, "status": "success"}
        )

    @mcp_server.tool(
        name="find",
        description="Search for files matching a pattern in directory",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def find(pattern: str, **kwargs) -> Dict[str, Any]:
        """Search for files matching a pattern in directory

        Args:
            pattern: File pattern to search for (e.g., '*.py', 'test*', 'config.json')
            **kwargs: Search options
                - directory: Directory to search in (defaults to current directory '.')
                - recursive: Search recursively in subdirectories
                - case_sensitive: Case-sensitive matching
                - max_depth: Maximum directory depth to search

        Examples:
            find(pattern="*.py")
            find(pattern="config.*", directory="/etc", recursive=True)
            find(pattern="TEST*", case_sensitive=False, max_depth=3)
        """
        directory = kwargs.get("directory", ".")
        result = await orchestrator.execute_tool("search_files", {"pattern": pattern, "directory": directory, **kwargs})
        return (
            result if isinstance(result, dict) else {"matches": str(result), "pattern": pattern, "directory": directory}
        )

    @mcp_server.tool(
        name="sysinfo",
        description="Get system information (OS, CPU, memory, disk space)",
        annotations=ToolAnnotations(category="system"),
    )
    async def sysinfo(**kwargs) -> Dict[str, Any]:
        """Get system information (OS, CPU, memory, disk space)

        Returns detailed system information including operating system, CPU details, memory usage, and disk space.
        """
        result = await orchestrator.execute_tool("get_system_info", {})
        return result if isinstance(result, dict) else {"system_info": str(result), "timestamp": "now"}

    @mcp_server.tool(
        name="init",
        description="Create a new project with multiple components (frontend, backend, database)",
        annotations=ToolAnnotations(category="project_management"),
    )
    async def init(
        name: str,
        description: str,
        components: List[Literal["frontend", "backend", "database", "api", "mobile"]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a new project with multiple components (frontend, backend, database)

        Args:
            name: Project name (e.g., 'my-web-app', 'data-pipeline')
            description: Project description (e.g., 'E-commerce website with React frontend')
            components: List of components to create (e.g., ['frontend', 'backend', 'database'])
            **kwargs: Additional project options
                - template: Project template to use
                - framework: Preferred framework (React, Vue, etc.)
                - database_type: Database type (PostgreSQL, MongoDB, etc.)

        Examples:
            init(name="ecommerce-app", description="Online store", components=["frontend", "backend", "database"])
            init(name="api-service", description="REST API", components=["backend"], framework="FastAPI", database_type="PostgreSQL")
        """
        result = await orchestrator.execute_tool(
            "create_project", {"name": name, "description": description, "components": components, **kwargs}
        )
        return (
            result
            if isinstance(result, dict)
            else {"name": name, "description": description, "components": components, "result": str(result)}
        )

    @mcp_server.tool(
        name="delegate",
        description="Distribute a task to specialized agents (frontend, backend, devops, etc.)",
        annotations=ToolAnnotations(category="orchestration"),
    )
    async def delegate(task_description: str, **kwargs) -> Dict[str, Any]:
        """Distribute a task to specialized agents (frontend, backend, devops, etc.)

        Args:
            task_description: Description of the task to distribute
            **kwargs: Delegation options
                - agent_type: Type of agent to assign task to (defaults to 'auto')
                - priority: Task priority (low, medium, high, urgent)
                - deadline: Task deadline
                - requirements: Specific requirements or constraints

        Examples:
            delegate(task_description="Build user authentication system")
            delegate(task_description="Optimize database queries", agent_type="database", priority="high")

        Returns:
            Dict containing task assignment results and agent response

        Raises:
            Error if no suitable agent is available or task fails
        """
        agent_type = kwargs.get("agent_type", "auto")
        result = await orchestrator.execute_tool(
            "distribute_task", {"task_description": task_description, "agent_type": agent_type, **kwargs}
        )
        return result if isinstance(result, dict) else {"result": str(result)}

    @mcp_server.tool(
        name="which",
        description="Check if a shell command exists and is available",
        annotations=ToolAnnotations(category="system"),
    )
    async def which(command: str, **kwargs) -> Dict[str, Any]:
        """Check if a shell command exists and is available

        Args:
            command: Command name to check (e.g., 'python', 'git', 'node')
            **kwargs: Additional options
                - path: Custom PATH to search in
                - all_paths: Return all found paths, not just the first
                - version: Also get version information if available

        Examples:
            which(command="python")
            which(command="node", version=True)
            which(command="git", all_paths=True)

        Returns:
            Dict containing command path and availability status

        Raises:
            Error if command check fails
        """
        result = await orchestrator.execute_tool("check_command_exists", {"command": command})
        return (
            result
            if isinstance(result, dict)
            else {"command": command, "available": bool(result), "path": str(result) if result else None}
        )

    @mcp_server.tool(
        name="projects",
        description="Get project and agent status",
        annotations=ToolAnnotations(category="project_management"),
    )
    async def projects(**kwargs) -> Dict[str, Any]:
        """Get project and agent status

        Returns:
            Dict containing project status and agent information
        """
        result = await orchestrator.execute_tool("get_project_status", {})
        return result if isinstance(result, dict) else {"status": str(result)}

    @mcp_server.tool(
        name="slack", description="Send Slack notification", annotations=ToolAnnotations(category="integrations")
    )
    async def slack(message: str, **kwargs) -> Dict[str, Any]:
        """Send Slack notification

        Args:
            message: Message to send
            **kwargs: Additional parameters
                - channel: Slack channel (defaults to #general)
                - thread_ts: Thread timestamp for replies
                - attachments: Message attachments

        Examples:
            slack(message="Deployment completed successfully")
            slack(message="Alert: High CPU usage", channel="#alerts")
            slack(message="Reply message", channel="#dev", thread_ts="1234567890.123")

        Returns:
            Dict containing message status and channel info
        """
        channel = kwargs.get("channel", "#general")
        result = await orchestrator.execute_tool("slack_notify", {"message": message, "channel": channel, **kwargs})
        return result if isinstance(result, dict) else {"message": message, "channel": channel, "status": "sent"}

    @mcp_server.tool(
        name="git",
        description="Perform Git operations (clone, pull, push, commit, branch)",
        annotations=ToolAnnotations(category="integrations"),
    )
    async def git(
        operation: Literal["clone", "pull", "push", "commit", "branch", "status", "add"], **kwargs
    ) -> Dict[str, Any]:
        """Perform Git operations (clone, pull, push, commit, branch)

        Args:
            operation: Git operation to perform
            **kwargs: Operation-specific parameters
                - repo_path: Repository path (defaults to current directory)
                - branch: Git branch (defaults to main)
                - Additional operation-specific parameters

        Examples:
            git(operation="status")
            git(operation="commit", message="Update feature", repo_path="/path/to/repo")
            git(operation="clone", url="https://github.com/user/repo.git", branch="develop")

        Returns:
            Dict containing git operation results
        """
        repo_path = kwargs.get("repo_path", ".")
        branch = kwargs.get("branch", "main")
        result = await orchestrator.execute_tool(
            "git_operation", {"operation": operation, "repo_path": repo_path, "branch": branch, **kwargs}
        )
        return result if isinstance(result, dict) else {"operation": operation, "result": str(result)}

    @mcp_server.tool(
        name="jenkins", description="Trigger Jenkins build", annotations=ToolAnnotations(category="integrations")
    )
    async def jenkins(job_name: str, **kwargs) -> Dict[str, Any]:
        """Trigger Jenkins build

        Args:
            job_name: Jenkins job name to trigger
            **kwargs: Build parameters and options

        Examples:
            jenkins(job_name="deploy-app")
            jenkins(job_name="build-feature", branch="develop", environment="staging")

        Returns:
            Dict containing build trigger status and job info
        """
        result = await orchestrator.execute_tool("jenkins_trigger", {"job_name": job_name, "parameters": kwargs})
        return (
            result if isinstance(result, dict) else {"job_name": job_name, "status": "triggered", "result": str(result)}
        )

    @mcp_server.tool(
        name="youtrack", description="Create YouTrack issue", annotations=ToolAnnotations(category="integrations")
    )
    async def youtrack(title: str, description: str, **kwargs) -> Dict[str, Any]:
        """Create YouTrack issue

        Args:
            title: Issue title
            description: Issue description
            **kwargs: Additional parameters
                - project: YouTrack project (defaults to 'default')
                - assignee: Issue assignee
                - priority: Issue priority
                - tags: Issue tags

        Examples:
            youtrack(title="Bug: Login fails", description="User cannot login with valid credentials")
            youtrack(title="Feature: Add dark mode", description="Implement dark theme", project="UI", priority="High")

        Returns:
            Dict containing issue creation status and issue ID
        """
        project = kwargs.get("project", "default")
        result = await orchestrator.execute_tool(
            "youtrack_issue", {"title": title, "description": description, "project": project, **kwargs}
        )
        return (
            result
            if isinstance(result, dict)
            else {"title": title, "project": project, "status": "created", "result": str(result)}
        )

    @mcp_server.tool(
        name="postgres", description="Execute PostgreSQL query", annotations=ToolAnnotations(category="integrations")
    )
    async def postgres(query: str, **kwargs) -> Dict[str, Any]:
        """Execute PostgreSQL query

        Args:
            query: SQL query to execute
            **kwargs: Additional parameters
                - database: Database name (defaults to 'default')
                - timeout: Query timeout in seconds
                - params: Query parameters for prepared statements

        Examples:
            postgres(query="SELECT * FROM users LIMIT 10")
            postgres(query="SELECT * FROM orders WHERE status = %s", params=["pending"], database="ecommerce")

        Returns:
            Dict containing query results and execution info
        """
        database = kwargs.get("database", "default")
        result = await orchestrator.execute_tool("postgres_query", {"query": query, "database": database, **kwargs})
        return result if isinstance(result, dict) else {"query": query, "database": database, "result": str(result)}

    @mcp_server.tool(
        name="aws",
        description="Execute AWS operation (EC2, S3, Lambda, RDS, etc.)",
        annotations=ToolAnnotations(category="integrations"),
    )
    async def aws(
        service: Literal["ec2", "s3", "lambda", "rds", "iam", "cloudformation"], operation: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute AWS operation (EC2, S3, Lambda, RDS, etc.)

        Args:
            service: AWS service to use
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Examples:
            aws(service="s3", operation="list_buckets")
            aws(service="ec2", operation="describe_instances", region="us-west-2")

        Returns:
            Dict containing AWS operation results
        """
        result = await orchestrator.execute_tool(
            "aws_operation", {"service": service, "operation": operation, "parameters": kwargs}
        )
        return (
            result if isinstance(result, dict) else {"service": service, "operation": operation, "result": str(result)}
        )

    @mcp_server.tool(
        name="bambu", description="Start Bambu Lab 3D print", annotations=ToolAnnotations(category="integrations")
    )
    async def bambu(file_path: str, **kwargs) -> Dict[str, Any]:
        """Start Bambu Lab 3D print

        Args:
            file_path: Path to 3D model file (.3mf, .gcode)
            **kwargs: Additional parameters
                - printer_id: Printer ID (defaults to 'default')
                - priority: Print job priority
                - filament_type: Filament material type
                - bed_temp: Bed temperature override

        Examples:
            bambu(file_path="/path/to/model.3mf")
            bambu(file_path="/path/to/part.gcode", printer_id="X1-Carbon-01", priority="high")

        Returns:
            Dict containing print job status and printer info
        """
        printer_id = kwargs.get("printer_id", "default")
        result = await orchestrator.execute_tool(
            "bambu_print", {"file_path": file_path, "printer_id": printer_id, **kwargs}
        )
        return (
            result
            if isinstance(result, dict)
            else {"file_path": file_path, "printer_id": printer_id, "status": "started", "result": str(result)}
        )

    @mcp_server.tool(
        name="websearch",
        description="Search the web for information",
        annotations=ToolAnnotations(category="integrations"),
    )
    async def websearch(query: str, **kwargs) -> Dict[str, Any]:
        """Search the web for information

        Args:
            query: Search query
            **kwargs: Search options
                - num_results: Number of results to return (defaults to 10)
                - language: Search language (e.g., 'en', 'es', 'fr')
                - region: Search region/country code
                - safe_search: Safe search level (strict, moderate, off)
                - time_range: Time range filter (day, week, month, year)

        Examples:
            websearch(query="Python tutorials")
            websearch(query="machine learning", num_results=20, language="en")
            websearch(query="latest news", time_range="day", safe_search="strict")

        Returns:
            Dict containing search results and metadata
        """
        num_results = kwargs.get("num_results", 10)
        result = await orchestrator.execute_tool("web_search", {"query": query, "num_results": num_results, **kwargs})
        return (
            result if isinstance(result, dict) else {"query": query, "num_results": num_results, "results": str(result)}
        )

    @mcp_server.tool(
        name="api",
        description="Make HTTP API requests (GET, POST, PUT, DELETE)",
        annotations=ToolAnnotations(category="integrations"),
    )
    async def api(url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP API requests (GET, POST, PUT, DELETE)

        Args:
            url: API endpoint URL
            **kwargs: Request parameters
                - method: HTTP method (defaults to GET)
                - headers: HTTP headers
                - data: Request data/body
                - timeout: Request timeout
                - auth: Authentication credentials

        Examples:
            api(url="https://api.example.com/users")
            api(url="https://api.example.com/users", method="POST", data={"name": "John"})
            api(url="https://api.example.com/secure", headers={"Authorization": "Bearer token"})

        Returns:
            Dict containing API response and metadata
        """
        method = kwargs.get("method", "GET")
        headers = kwargs.get("headers", {})
        data = kwargs.get("data", {})
        result = await orchestrator.execute_tool(
            "api_request", {"url": url, "method": method, "headers": headers, "data": data, **kwargs}
        )
        return (
            result
            if isinstance(result, dict)
            else {"url": url, "method": method, "status": "completed", "response": str(result)}
        )

    # 🎬 Multimedia Processing Tools
    @mcp_server.tool(
        name="ffmpeg",
        description="Video processing with FFmpeg (convert, compress, extract audio, generate thumbnails)",
        annotations=ToolAnnotations(category="multimedia"),
    )
    async def ffmpeg(
        operation: Literal[
            "convert", "compress", "extract_audio", "generate_thumbnail", "get_info", "trim", "merge", "add_watermark"
        ],
        **kwargs,
    ) -> Dict[str, Any]:
        """Video processing with FFmpeg

        Args:
            operation: FFmpeg operation to perform
            **kwargs: Operation-specific parameters
                - input_path: Input video file path (required)
                - output_path: Output file path (auto-generated if not provided)
                - Additional operation-specific options

        Examples:
            ffmpeg(operation="convert", input_path="/path/to/video.mp4", output_path="/path/to/output.avi")
            ffmpeg(operation="compress", input_path="/path/to/video.mp4", quality=0.5)
            ffmpeg(operation="extract_audio", input_path="/path/to/video.mp4")
        """
        from integrations.multimedia import ffmpeg_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await ffmpeg_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="opencv",
        description="Computer vision with OpenCV (detect faces, extract frames, motion detection)",
        annotations=ToolAnnotations(category="multimedia"),
    )
    async def opencv(
        operation: Literal["detect_faces", "detect_objects", "extract_frames", "motion_detection", "blur_faces"],
        **kwargs,
    ) -> Dict[str, Any]:
        """Computer vision with OpenCV

        Args:
            operation: OpenCV operation to perform
            **kwargs: Operation-specific parameters
                - image_path: Input image/video file path (required)
                - Additional operation-specific options

        Examples:
            opencv(operation="detect_faces", image_path="/path/to/image.jpg")
            opencv(operation="blur_faces", image_path="/path/to/image.jpg", blur_factor=10)
        """
        from integrations.multimedia import opencv_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await opencv_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="whisper",
        description="Speech-to-text transcription with OpenAI Whisper",
        annotations=ToolAnnotations(category="multimedia"),
    )
    async def whisper(operation: Literal["transcribe", "translate"], **kwargs) -> Dict[str, Any]:
        """Speech-to-text with Whisper

        Args:
            operation: Whisper operation (transcribe or translate)
            **kwargs: Operation-specific parameters
                - audio_path: Audio file path (required)
                - language: Source language (auto-detect if None)

        Examples:
            whisper(operation="transcribe", audio_path="/path/to/audio.mp3")
            whisper(operation="translate", audio_path="/path/to/audio.mp3", language="es")
        """
        from integrations.multimedia import whisper_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await whisper_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="elevenlabs",
        description="Text-to-speech with ElevenLabs AI voices",
        annotations=ToolAnnotations(category="multimedia"),
    )
    async def elevenlabs(
        operation: Literal["text_to_speech", "list_voices", "clone_voice"], **kwargs
    ) -> Dict[str, Any]:
        """Text-to-speech with ElevenLabs

        Args:
            operation: ElevenLabs operation
            **kwargs: Operation-specific parameters
                - text: Text to convert to speech (required for text_to_speech)
                - voice_id: Voice ID to use
                - stability: Voice stability (0.0-1.0)
                - similarity_boost: Similarity boost (0.0-1.0)
                - output_format: Audio output format

        Examples:
            elevenlabs(operation="list_voices")
            elevenlabs(operation="text_to_speech", text="Hello world", voice_id="21m00Tcm4TlvDq8ikWAM")
            elevenlabs(operation="clone_voice", voice_id="custom", stability=0.8)
        """
        from integrations.multimedia import elevenlabs_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await elevenlabs_integration.execute_operation(operation, **final_kwargs)

    # 🤖 AI/ML Platform Tools
    @mcp_server.tool(
        name="huggingface",
        description="HuggingFace models and inference (text generation, classification, embeddings)",
        annotations=ToolAnnotations(category="ai_ml"),
    )
    async def huggingface(
        operation: Literal[
            "generate_text", "classify_text", "embed_text", "search_models", "get_model_info", "generate_image"
        ],
        **kwargs,
    ) -> Dict[str, Any]:
        """HuggingFace AI models and inference

        Args:
            operation: HuggingFace operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.ai_ml import huggingface_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await huggingface_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="openai",
        description="OpenAI API (ChatGPT, DALL-E, Whisper, embeddings)",
        annotations=ToolAnnotations(category="ai_ml"),
    )
    async def openai(
        operation: Literal["chat", "complete", "generate_image", "transcribe", "embed"], **kwargs
    ) -> Dict[str, Any]:
        """OpenAI API access

        Args:
            operation: OpenAI operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.ai_ml import openai_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await openai_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="anthropic",
        description="Anthropic Claude API for advanced AI conversations",
        annotations=ToolAnnotations(category="ai_ml"),
    )
    async def anthropic(operation: Literal["chat"], **kwargs) -> Dict[str, Any]:
        """Anthropic Claude API

        Args:
            operation: Anthropic operation
            **kwargs: Operation-specific parameters
                - For chat: messages (required), model, max_tokens

        Examples:
            anthropic(operation="chat", messages=[{"role": "user", "content": "Hello"}], model="claude-3")
        """
        from integrations.ai_ml import anthropic_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await anthropic_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="replicate",
        description="Run AI models on Replicate cloud platform",
        annotations=ToolAnnotations(category="ai_ml"),
    )
    async def replicate(operation: Literal["run_model", "get_prediction", "list_models"], **kwargs) -> Dict[str, Any]:
        """Replicate AI model platform

        Args:
            operation: Replicate operation
            **kwargs: Operation-specific parameters
                - For run_model: model (required), input_data
                - For get_prediction: prediction_id
                - For list_models: (no additional parameters)

        Examples:
            replicate(operation="run_model", model="stability-ai/stable-diffusion", input_data={"prompt": "A cat"})
            replicate(operation="list_models")
        """
        from integrations.ai_ml import replicate_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await replicate_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="ollama",
        description="Execute local Ollama AI operations (generate, chat, embed, list_models, pull_model, show_model)",
        annotations=ToolAnnotations(category="ai_ml"),
    )
    async def ollama(
        operation: Literal["generate", "chat", "embed", "list_models", "pull_model", "show_model"], **kwargs
    ) -> Dict[str, Any]:
        """Ollama local AI model server operations

        Args:
            operation: Ollama operation (generate, chat, embed, list_models, pull_model, show_model)
            **kwargs: Operation-specific parameters (supports both dict and JSON string via 'kwargs' key)
                - For generate: model, prompt, options (temperature, max_tokens, etc.)
                - For chat: model, messages, options
                - For embed: model, input_text, options
                - For list_models: (no additional parameters)
                - For pull_model: model_name
                - For show_model: model_name

        Examples:
            # Direct kwargs (Python/dict mode):
            ollama(operation="generate", model="llama3", prompt="Hello", options={"temperature": 0.7})
            # JSON string mode (Claude MCP):
            ollama(operation="generate", kwargs='{"model": "llama3", "prompt": "Hello"}')
        """
        from integrations.ai_ml import ollama_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await ollama_integration.execute_operation(operation, **final_kwargs)

    # 🌐 Productivity & Communication Tools
    @mcp_server.tool(
        name="discord",
        description="Discord bot and webhook messaging",
        annotations=ToolAnnotations(category="productivity"),
    )
    async def discord(
        operation: Literal["send_message", "send_webhook", "create_channel", "get_guilds"], **kwargs
    ) -> Dict[str, Any]:
        """Discord integration for messaging and automation

        Args:
            operation: Discord operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.productivity import discord_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await discord_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="email", description="Send emails via SMTP", annotations=ToolAnnotations(category="productivity")
    )
    async def email(
        operation: Literal["send", "send_html"], to_email: str, subject: str, body: str, **kwargs
    ) -> Dict[str, Any]:
        """Email integration for sending messages

        Args:
            operation: Email operation
            to_email: Recipient email address
            subject: Email subject
            body: Email body content
            **kwargs: Additional email options
                - from_email: Sender email (optional)
                - cc: CC recipients
                - bcc: BCC recipients
                - attachments: Email attachments
                - priority: Email priority (high, normal, low)

        Examples:
            email(operation="send", to_email="user@example.com", subject="Hello", body="Message")
            email(operation="send_html", to_email="user@example.com", subject="Newsletter",
                  body="<h1>HTML content</h1>", cc=["cc@example.com"])
        """
        from integrations.productivity import email_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await email_integration.execute_operation(
            operation, to_email=to_email, subject=subject, body=body, **final_kwargs
        )

    @mcp_server.tool(
        name="notion",
        description="Notion workspace and database management",
        annotations=ToolAnnotations(category="productivity"),
    )
    async def notion(
        operation: Literal["create_page", "update_page", "query_database", "create_database"], **kwargs
    ) -> Dict[str, Any]:
        """Notion integration for workspace management

        Args:
            operation: Notion operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.productivity import notion_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await notion_integration.execute_operation(operation, **final_kwargs)

    # 🛠 Developer Tools & Infrastructure
    @mcp_server.tool(
        name="docker",
        description="Docker container management (run, build, deploy, monitor)",
        annotations=ToolAnnotations(category="devtools"),
    )
    async def docker(
        operation: Literal["list_containers", "run_container", "build_image", "get_logs", "exec_command"], **kwargs
    ) -> Dict[str, Any]:
        """Docker container management

        Args:
            operation: Docker operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.devtools import docker_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await docker_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="kubernetes",
        description="Kubernetes cluster management (pods, services, deployments)",
        annotations=ToolAnnotations(category="devtools"),
    )
    async def kubernetes(
        operation: Literal["get_pods", "get_services", "apply_manifest", "delete_resource", "scale_deployment"],
        **kwargs,
    ) -> Dict[str, Any]:
        """Kubernetes cluster management

        Args:
            operation: Kubernetes operation to perform
            **kwargs: Operation-specific parameters
        """
        from integrations.devtools import kubernetes_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await kubernetes_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="terraform",
        description="Infrastructure as Code with Terraform (plan, apply, destroy)",
        annotations=ToolAnnotations(category="devtools"),
    )
    async def terraform(
        operation: Literal["init", "plan", "apply", "destroy", "validate", "show"],
        working_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Terraform infrastructure management

        Args:
            operation: Terraform operation to perform
            working_dir: Terraform working directory
            **kwargs: Operation-specific parameters
        """
        from integrations.devtools import terraform_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await terraform_integration.execute_operation(operation, working_dir=working_dir, **final_kwargs)

    @mcp_server.tool(
        name="status",
        description="Get orchestrator and agent system status",
        annotations=ToolAnnotations(category="system"),
    )
    async def status(**kwargs) -> Dict[str, Any]:
        """Get orchestrator and agent system status

        Returns:
            Dict containing system status and agent health information
        """
        result = await orchestrator.execute_tool("get_system_status", {})
        return result if isinstance(result, dict) else {"status": str(result)}

    @mcp_server.tool(
        name="agents",
        description="List all available agents and their specializations",
        annotations=ToolAnnotations(category="system"),
    )
    async def agents(**kwargs) -> Dict[str, Any]:
        """List all available agents and their specializations

        Returns:
            Dict containing agent list and their capabilities
        """
        result = await orchestrator.execute_tool("list_agents", {})
        return result if isinstance(result, dict) else {"agents": str(result)}

    @mcp_server.tool(
        name="task",
        description="Create a new task with title, description and priority",
        annotations=ToolAnnotations(category="task_management"),
    )
    async def task(title: str, description: str, **kwargs) -> Dict[str, Any]:
        """Create a new task with title, description and priority

        Args:
            title: Task title (e.g., 'Fix login bug', 'Add user dashboard')
            description: Detailed task description
            **kwargs: Task options
                - priority: Task priority level (defaults to 'medium')
                - assignee: Task assignee
                - due_date: Task due date
                - tags: Task tags/labels
                - project: Project association

        Examples:
            task(title="Fix login bug", description="Users cannot authenticate")
            task(title="Add dashboard", description="Create user dashboard",
                 priority="high", assignee="john", due_date="2024-01-15")

        Returns:
            Dict containing task ID and creation status

        Raises:
            Error if task creation fails or invalid parameters provided
        """
        priority = kwargs.get("priority", "medium")
        result = await orchestrator.execute_tool(
            "create_task", {"title": title, "description": description, "priority": priority, **kwargs}
        )
        return result if isinstance(result, dict) else {"result": str(result)}

    # ============================================================================
    # 🏢 ENTERPRISE & BUSINESS INTEGRATIONS
    # ============================================================================

    @mcp_server.tool(
        name="salesforce",
        description="Execute Salesforce CRM operations (create_lead, update_opportunity, query_records, create_account)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def salesforce(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Salesforce CRM operations

        Args:
            operation: Operation to perform (create_lead, update_opportunity, query_records, create_account)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import salesforce_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await salesforce_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="hubspot",
        description="Execute HubSpot CRM operations (create_contact, create_deal, get_contacts)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def hubspot(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute HubSpot CRM operations

        Args:
            operation: Operation to perform (create_contact, create_deal, get_contacts)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import hubspot_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await hubspot_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="stripe",
        description="Execute Stripe payment operations (create_payment_intent, create_customer, list_charges, get_balance)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def stripe(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Stripe payment operations

        Args:
            operation: Operation to perform (create_payment_intent, create_customer, list_charges, get_balance)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import stripe_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await stripe_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="pipedrive",
        description="Execute Pipedrive CRM operations (create_deal, create_person, get_deals, update_deal)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def pipedrive(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Pipedrive CRM operations

        Args:
            operation: Operation to perform (create_deal, create_person, get_deals, update_deal)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import pipedrive_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await pipedrive_integration.execute_operation(operation, **final_kwargs)

    # Project Management Tools
    @mcp_server.tool(
        name="jira",
        description="Execute Jira project management operations (create_issue, get_issue, search_issues, update_issue, get_projects)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def jira(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Jira project management operations

        Args:
            operation: Operation to perform (create_issue, get_issue, search_issues, update_issue, get_projects)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import jira_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await jira_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="asana",
        description="Execute Asana task management operations (create_task, get_tasks, update_task, get_projects)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def asana(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Asana task management operations

        Args:
            operation: Operation to perform (create_task, get_tasks, update_task, get_projects)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import asana_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await asana_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="trello",
        description="Execute Trello board management operations (create_card, get_boards, get_lists, move_card)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def trello(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Trello board management operations

        Args:
            operation: Operation to perform (create_card, get_boards, get_lists, move_card)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import trello_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await trello_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="linear",
        description="Execute Linear issue tracking operations (create_issue, get_issues, update_issue)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def linear(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Linear issue tracking operations

        Args:
            operation: Operation to perform (create_issue, get_issues, update_issue)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import linear_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await linear_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="monday",
        description="Execute Monday.com work management operations (create_item, get_boards, update_item)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def monday(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Monday.com work management operations

        Args:
            operation: Operation to perform (create_item, get_boards, update_item)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import monday_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await monday_integration.execute_operation(operation, **final_kwargs)

    # HR & Operations Tools
    @mcp_server.tool(
        name="workday",
        description="Execute Workday HR management operations (get_workers, get_worker, get_organizations, get_time_off)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def workday(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Workday HR management operations

        Args:
            operation: Operation to perform (get_workers, get_worker, get_organizations, get_time_off)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import workday_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await workday_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="bamboohr",
        description="Execute BambooHR operations (get_employees, get_employee, get_time_off_requests, update_employee)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def bamboohr(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute BambooHR HR information system operations

        Args:
            operation: Operation to perform (get_employees, get_employee, get_time_off_requests, update_employee)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import bamboohr_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await bamboohr_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="toggl",
        description="Execute Toggl time tracking operations (start_timer, stop_timer, get_time_entries, create_project)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def toggl(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Toggl time tracking operations

        Args:
            operation: Operation to perform (start_timer, stop_timer, get_time_entries, create_project)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import toggl_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await toggl_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="harvest",
        description="Execute Harvest time tracking and invoicing operations (create_time_entry, get_time_entries, get_projects, create_invoice)",
        annotations=ToolAnnotations(category="enterprise"),
    )
    async def harvest(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Harvest time tracking and invoicing operations

        Args:
            operation: Operation to perform (create_time_entry, get_time_entries, get_projects, create_invoice)
            **kwargs: Operation-specific parameters
        """
        from integrations.enterprise import harvest_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await harvest_integration.execute_operation(operation, **final_kwargs)

    # ============================================================================
    # 🔒 SECURITY & COMPLIANCE INTEGRATIONS
    # ============================================================================

    @mcp_server.tool(
        name="vault",
        description="Execute HashiCorp Vault operations (read_secret, write_secret, delete_secret, list_secrets)",
        annotations=ToolAnnotations(category="security"),
    )
    async def vault(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute HashiCorp Vault secret management operations

        Args:
            operation: Operation to perform (read_secret, write_secret, delete_secret, list_secrets)
            **kwargs: Operation-specific parameters
        """
        from integrations.security import vault_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await vault_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="okta",
        description="Execute Okta identity management operations (list_users, create_user, deactivate_user, list_groups)",
        annotations=ToolAnnotations(category="security"),
    )
    async def okta(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Okta identity management operations

        Args:
            operation: Operation to perform (list_users, create_user, deactivate_user, list_groups)
            **kwargs: Operation-specific parameters
        """
        from integrations.security import okta_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await okta_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="auth0",
        description="Execute Auth0 identity management operations (list_users, create_user, update_user, get_logs)",
        annotations=ToolAnnotations(category="security"),
    )
    async def auth0(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Auth0 identity management operations

        Args:
            operation: Operation to perform (list_users, create_user, update_user, get_logs)
            **kwargs: Operation-specific parameters
        """
        from integrations.security import auth0_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await auth0_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="snyk",
        description="Execute Snyk vulnerability scanning operations (test_project, list_projects, get_vulnerabilities, monitor_project)",
        annotations=ToolAnnotations(category="security"),
    )
    async def snyk(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Snyk vulnerability scanning operations

        Args:
            operation: Operation to perform (test_project, list_projects, get_vulnerabilities, monitor_project)
            **kwargs: Operation-specific parameters
        """
        from integrations.security import snyk_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await snyk_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="sonarqube",
        description="Execute SonarQube code quality operations (list_projects, get_project_metrics, get_issues, create_project)",
        annotations=ToolAnnotations(category="security"),
    )
    async def sonarqube(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute SonarQube code quality and security analysis operations

        Args:
            operation: Operation to perform (list_projects, get_project_metrics, get_issues, create_project)
            **kwargs: Operation-specific parameters
        """
        from integrations.security import sonarqube_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await sonarqube_integration.execute_operation(operation, **final_kwargs)

    # ============================================================================
    # 🎮 SPECIALIZED & FUN INTEGRATIONS
    # ============================================================================

    @mcp_server.tool(
        name="steam",
        description="Execute Steam gaming platform operations (get_player_summaries, get_owned_games, get_friends_list, get_app_details)",
        annotations=ToolAnnotations(category="gaming"),
    )
    async def steam(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Steam gaming platform operations

        Args:
            operation: Operation to perform (get_player_summaries, get_owned_games, get_friends_list, get_app_details)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import steam_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await steam_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="twitch",
        description="Execute Twitch streaming operations (get_streams, get_users, get_games, search_channels)",
        annotations=ToolAnnotations(category="gaming"),
    )
    async def twitch(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Twitch streaming platform operations

        Args:
            operation: Operation to perform (get_streams, get_users, get_games, search_channels)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import twitch_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await twitch_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="arduino",
        description="Execute Arduino microcontroller operations (send_command, read_data, upload_sketch, get_board_info)",
        annotations=ToolAnnotations(category="iot"),
    )
    async def arduino(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Arduino microcontroller operations

        Args:
            operation: Operation to perform (send_command, read_data, upload_sketch, get_board_info)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import arduino_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await arduino_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="mqtt",
        description="Execute MQTT message broker operations (publish, subscribe, get_broker_info)",
        annotations=ToolAnnotations(category="iot"),
    )
    async def mqtt(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute MQTT message broker operations

        Args:
            operation: Operation to perform (publish, subscribe, get_broker_info)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import mqtt_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await mqtt_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="ethereum",
        description="Execute Ethereum blockchain operations (get_balance, get_transaction, send_transaction, call_contract, get_gas_price)",
        annotations=ToolAnnotations(category="blockchain"),
    )
    async def ethereum(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Ethereum blockchain operations

        Args:
            operation: Operation to perform (get_balance, get_transaction, send_transaction, call_contract, get_gas_price)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import ethereum_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await ethereum_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="bitcoin",
        description="Execute Bitcoin blockchain operations (get_address_info, get_transaction, get_block, get_mempool_info)",
        annotations=ToolAnnotations(category="blockchain"),
    )
    async def bitcoin(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Bitcoin blockchain operations

        Args:
            operation: Operation to perform (get_address_info, get_transaction, get_block, get_mempool_info)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import bitcoin_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await bitcoin_integration.execute_operation(operation, **final_kwargs)

    @mcp_server.tool(
        name="nft",
        description="Execute NFT operations (get_nft_collection, get_nft_metadata, list_user_nfts, get_floor_price)",
        annotations=ToolAnnotations(category="blockchain"),
    )
    async def nft(operation: str, **kwargs) -> Dict[str, Any]:
        """Execute NFT and digital collectibles operations

        Args:
            operation: Operation to perform (get_nft_collection, get_nft_metadata, list_user_nfts, get_floor_price)
            **kwargs: Operation-specific parameters
        """
        from integrations.specialized import nft_integration

        final_kwargs = process_kwargs_dual_mode(**kwargs)
        return await nft_integration.execute_operation(operation, **final_kwargs)

    # ============================================================================
    # 🎨 DESIGN & DEVELOPMENT TOOLS
    # ============================================================================

    @mcp_server.tool(
        name="sketch_transpile",
        description="Convert Sketch design files to SwiftUI code using the sketch-transpiler",
        annotations=ToolAnnotations(category="design"),
    )
    async def sketch_transpile(sketch_file_path: str, **kwargs) -> Dict[str, Any]:
        """Convert Sketch design files to SwiftUI code

        Args:
            sketch_file_path: Path to the .sketch file to transpile
            **kwargs: Additional parameters
                - output_dir: Directory to save generated Swift files (defaults to './swift-output')
                - verbose: Enable verbose output (defaults to False)
                - save_files: Whether to save files to disk (defaults to True)

        Examples:
            sketch_transpile(sketch_file_path="/path/to/design.sketch")
            sketch_transpile(sketch_file_path="./ui-design.sketch", output_dir="./generated", verbose=True)
            sketch_transpile(sketch_file_path="design.sketch", save_files=False)  # Returns Swift code without saving

        Returns:
            Dict containing Swift code files and generation status
        """
        final_kwargs = process_kwargs_dual_mode(**kwargs)
        result = await orchestrator.execute_tool(
            "sketch_transpile", {"sketch_file_path": sketch_file_path, **final_kwargs}
        )
        return result if isinstance(result, dict) else {"status": "completed", "result": str(result)}

    return mcp_server


async def create_stdio_mcp_server() -> FastMCP:
    """Create lightweight stdio MCP server with basic tools (alternative to full integration system)"""
    mcp_server = FastMCP("wand-basic")

    # Add basic tools with proper category annotations
    @mcp_server.tool(
        name="execute_command",
        description="Execute shell command via orchestrator",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def execute_command(command: str, **kwargs) -> str:
        """Execute shell command via orchestrator

        Args:
            command: Command to execute
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator("execute_command", {"command": command, **kwargs})
        return str(result)

    @mcp_server.tool(
        name="read_file",
        description="Read file contents via orchestrator",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def read_file(file_path: str, **kwargs) -> str:
        """Read file contents via orchestrator

        Args:
            file_path: Path to file to read
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator("read_file", {"file_path": file_path, **kwargs})
        return str(result)

    @mcp_server.tool(
        name="write_file",
        description="Write file contents via orchestrator",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def write_file(file_path: str, content: str, **kwargs) -> str:
        """Write file contents via orchestrator

        Args:
            file_path: Path to file to write
            content: Content to write
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator(
            "write_file", {"file_path": file_path, "content": content, **kwargs}
        )
        return str(result)

    @mcp_server.tool(
        name="list_directory",
        description="List directory contents via orchestrator",
        annotations=ToolAnnotations(category="file_operations"),
    )
    async def list_directory(directory: str = ".", **kwargs) -> str:
        """List directory contents via orchestrator

        Args:
            directory: Directory to list (default: current directory)
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator("list_directory", {"directory": directory, **kwargs})
        return str(result)

    @mcp_server.tool(
        name="get_system_info",
        description="Get system information via orchestrator",
        annotations=ToolAnnotations(category="system"),
    )
    async def get_system_info(**kwargs) -> str:
        """Get system information via orchestrator

        Args:
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator("get_system_info", {**kwargs})
        return str(result)

    @mcp_server.tool(
        name="create_project",
        description="Create multi-component project via orchestrator",
        annotations=ToolAnnotations(category="system"),
    )
    async def create_project(name: str, description: str, components: list, **kwargs) -> str:
        """Create multi-component project via orchestrator

        Args:
            name: Project name
            description: Project description
            components: List of components to create
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import execute_tool_via_orchestrator

        result = await execute_tool_via_orchestrator(
            "create_project", {"name": name, "description": description, "components": components, **kwargs}
        )
        return str(result)

    @mcp_server.tool(
        name="get_system_status",
        description="Get system status via orchestrator",
        annotations=ToolAnnotations(category="system"),
    )
    async def get_system_status(**kwargs) -> str:
        """Get system status via orchestrator

        Args:
            **kwargs: Additional parameters
        """
        from orchestrator.agent_orchestrator import get_orchestrator_status

        result = await get_orchestrator_status()
        return str(result)

    return mcp_server


async def run_stdio_mode():
    """Run stdio mode with orchestrator"""
    orchestrator = AgentOrchestrator()
    await orchestrator.start()
    logger.info("✓ Agent orchestrator started")

    try:
        # Create orchestrator-backed stdio MCP server
        mcp_server = await create_orchestrator_stdio_server(orchestrator)
        logger.info("✓ Orchestrator-backed stdio MCP server created")
        logger.info("📡 Ready for MCP clients via stdio transport")

        # Use the FastMCP run method (not async)
        mcp_server.run(transport="stdio")
    finally:
        await orchestrator.stop()
        logger.info("✓ Agent orchestrator stopped")


async def run_http_mode():
    """Run HTTP mode with orchestrator"""
    config = SimpleConfig()
    orchestrator = AgentOrchestrator()
    await orchestrator.start()
    logger.info("✓ Agent orchestrator started")

    try:
        # Create HTTP transport
        http_transport = MCPHttpTransport(orchestrator, config.server.version)
        logger.info("✓ HTTP MCP transport created")

        # Add SSE transport to the same app
        sse_transport = MCPSSETransport(orchestrator, config.server.version)
        sse_transport.setup_routes(http_transport.app)
        logger.info("✓ SSE transport added for Claude compatibility")

        # Start HTTP server
        port = getattr(config.server, 'mcp_http_port', 8001)
        logger.info(f"🌐 Starting HTTP MCP server on port {port}")
        logger.info(f"   - POST /mcp for standard MCP")
        logger.info(f"   - GET/POST /sse for Claude SSE")

        uvicorn_config = uvicorn.Config(http_transport.app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    finally:
        await orchestrator.stop()
        logger.info("✓ Agent orchestrator stopped")


def main():
    """Main entry point for the MCP multi-agent system"""
    logger.info("🚀 Starting MCP Multi-Agent System")
    logger.info("=" * 60)

    # Initialize config
    config = SimpleConfig()

    # Determine transport mode
    transport_mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport_mode == "stdio":
        logger.info("Starting stdio transport for MCP clients")

        # Initialize orchestrator and create MCP server in sync way
        async def setup_stdio():
            orchestrator = AgentOrchestrator()
            await orchestrator.start()
            logger.info("✓ Agent orchestrator started")

            # Create orchestrator-backed stdio MCP server
            return await create_orchestrator_stdio_server(orchestrator), orchestrator

        # Setup the MCP server
        mcp_server, orchestrator = asyncio.run(setup_stdio())
        logger.info("✓ Orchestrator-backed stdio MCP server created")
        logger.info("📡 Ready for MCP clients via stdio transport")

        try:
            # Now run FastMCP in non-async context
            mcp_server.run(transport="stdio")
        finally:
            # Cleanup orchestrator
            async def cleanup():
                await orchestrator.stop()
                logger.info("✓ Agent orchestrator stopped")

            asyncio.run(cleanup())

    elif transport_mode == "http":
        logger.info("Starting HTTP transport for OpenCode")
        asyncio.run(run_http_mode())

    else:
        logger.error(f"Unknown transport mode: {transport_mode}")
        logger.info("Usage: python wand.py [stdio|http]")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
