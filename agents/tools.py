#!/usr/bin/env python3
"""
Agent Tools - All 22 MCP tools for agent instances

Extracted from distributed_server.py to be used by individual agents.
Each agent instance will have access to all these tools.
"""

import asyncio
import glob
import json
import logging
import os
import platform
import subprocess

# Import execution backend and other dependencies
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from core.task_manager import TaskPriority
from tools.execution.factory import create_execution_backend

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)


class AgentTools:
    """Container for all agent tools"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config

        # Initialize execution backend with default config
        try:
            self.execution_backend = create_execution_backend("native", {})
        except Exception as e:
            logger.warning(f"Failed to create execution backend: {e}")
            self.execution_backend = None

    async def execute_command(
        self, command: str, timeout: int = 30, working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute shell command"""
        try:
            if self.execution_backend:
                from tools.execution.base import ExecutionConfig

                exec_config = ExecutionConfig(
                    command=command, timeout=timeout, working_directory=working_dir or os.getcwd()
                )
                result = await self.execution_backend.execute(exec_config)
                return {
                    "success": result.status.name == "SUCCESS",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                }
            else:
                # Fallback to subprocess
                process = await asyncio.create_subprocess_shell(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "exit_code": process.returncode,
                }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e), "stdout": "", "stderr": "", "exit_code": -1}

    async def read_file(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file contents"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            if not path.is_file():
                return {"success": False, "error": f"Path is not a file: {file_path}"}

            content = path.read_text(encoding=encoding)
            return {"success": True, "content": content, "size": len(content), "encoding": encoding}
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return {"success": False, "error": str(e)}

    async def write_file(self, file_path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
        """Write content to file"""
        try:
            path = Path(file_path)

            if create_dirs and path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding='utf-8')

            return {
                "success": True,
                "file_path": str(path),
                "size": len(content),
                "message": f"Successfully wrote {len(content)} characters to {file_path}",
            }
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return {"success": False, "error": str(e)}

    async def list_directory(self, directory: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
        """List directory contents"""
        try:
            path = Path(directory)
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            if not path.is_dir():
                return {"success": False, "error": f"Path is not a directory: {directory}"}

            items = []
            for item in path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue

                stat = item.stat()
                items.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else None,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

            return {
                "success": True,
                "directory": str(path.absolute()),
                "items": sorted(items, key=lambda x: (x["type"], x["name"])),
                "count": len(items),
            }
        except Exception as e:
            logger.error(f"Directory listing failed: {e}")
            return {"success": False, "error": str(e)}

    async def search_files(self, pattern: str, directory: str = ".", file_type: Optional[str] = None) -> Dict[str, Any]:
        """Search for files matching pattern"""
        try:
            search_path = Path(directory)
            if not search_path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            # Build glob pattern
            if file_type:
                glob_pattern = f"**/*.{file_type}"
            else:
                glob_pattern = f"**/*{pattern}*"

            matches = []
            for match in search_path.glob(glob_pattern):
                if match.is_file():
                    stat = match.stat()
                    matches.append(
                        {
                            "path": str(match),
                            "name": match.name,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )

            return {
                "success": True,
                "pattern": pattern,
                "directory": directory,
                "matches": matches,
                "count": len(matches),
            }
        except Exception as e:
            logger.error(f"File search failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                "success": True,
                "system": {
                    "platform": platform.system(),
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "hostname": platform.node(),
                    "processor": platform.processor(),
                },
                "resources": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": {
                        "total": psutil.virtual_memory().total,
                        "available": psutil.virtual_memory().available,
                        "percent": psutil.virtual_memory().percent,
                    },
                    "disk": {
                        "total": psutil.disk_usage('/').total,
                        "free": psutil.disk_usage('/').free,
                        "percent": psutil.disk_usage('/').percent,
                    },
                },
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"System info failed: {e}")
            return {"success": False, "error": str(e)}

    async def check_command_exists(self, command: str) -> Dict[str, Any]:
        """Check if command exists in system PATH"""
        try:
            import shutil

            path = shutil.which(command)

            return {
                "success": True,
                "command": command,
                "exists": path is not None,
                "path": path,
                "agent_id": self.agent_id,
            }
        except Exception as e:
            logger.error(f"Command check failed: {e}")
            return {"success": False, "error": str(e)}

    async def create_project(self, name: str, description: str, components: List[str]) -> Dict[str, Any]:
        """Create a new project with multiple components"""
        try:
            project_id = str(uuid.uuid4())

            # This would integrate with the task manager in a real implementation
            tasks = []
            for component in components:
                task_info = {
                    "id": str(uuid.uuid4()),
                    "title": f"{name} - {component.capitalize()} Development",
                    "description": f"Develop {component} for {name}",
                    "component": component,
                    "status": "pending",
                }
                tasks.append(task_info)

            return {
                "success": True,
                "project_id": project_id,
                "name": name,
                "description": description,
                "components": components,
                "tasks": tasks,
                "created_by": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                "success": True,
                "agent_id": self.agent_id,
                "status": "operational",
                "uptime": psutil.boot_time(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "processes": len(psutil.pids()),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"System status failed: {e}")
            return {"success": False, "error": str(e)}

    async def api_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Make HTTP API request"""
        try:
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(method.upper(), url, headers=headers or {}, json=data) as response:
                    response_text = await response.text()

                    try:
                        response_json = await response.json()
                    except BaseException:
                        response_json = None

                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "text": response_text,
                        "json": response_json,
                        "url": str(response.url),
                    }
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {"success": False, "error": str(e)}

    # Placeholder methods for integration tools
    async def slack_notify(self, channel: str, message: str, **kwargs) -> Dict[str, Any]:
        """Send Slack notification"""
        return {"success": False, "error": "Slack integration not configured"}

    async def git_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Perform Git operations"""
        return {"success": False, "error": "Git integration not implemented"}

    async def jenkins_trigger(self, job_name: str, **kwargs) -> Dict[str, Any]:
        """Trigger Jenkins build"""
        return {"success": False, "error": "Jenkins integration not configured"}

    async def youtrack_issue(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Create YouTrack issue"""
        return {"success": False, "error": "YouTrack integration not configured"}

    async def postgres_query(self, database: str, query: str, **kwargs) -> Dict[str, Any]:
        """Execute PostgreSQL query"""
        return {"success": False, "error": "PostgreSQL integration not configured"}

    async def aws_operation(self, service: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute AWS operation"""
        return {"success": False, "error": "AWS integration not configured"}

    async def bambu_print(self, printer_id: str, **kwargs) -> Dict[str, Any]:
        """Start Bambu Lab 3D print"""
        return {"success": False, "error": "Bambu Lab integration not configured"}

    async def web_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search the web"""
        return {"success": False, "error": "Web search integration not configured"}

    async def claude_api_call(
        self,
        message: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make API call to Claude for AI-powered tasks"""
        try:
            import os

            # Check for API key
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return {"success": False, "error": "ANTHROPIC_API_KEY environment variable not set"}

            # Import anthropic client
            try:
                import anthropic
            except ImportError:
                return {"success": False, "error": "anthropic package not installed. Run: pip install anthropic"}

            # Initialize client
            client = anthropic.Anthropic(api_key=api_key)

            # Prepare messages
            messages = [{"role": "user", "content": message}]

            # Prepare request parameters
            request_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            # Make API call
            logger.info(f"Making Claude API call with model {model}")
            response = client.messages.create(**request_params)

            # Extract response text
            response_text = ""
            for content in response.content:
                if content.type == "text":
                    response_text += content.text

            return {
                "success": True,
                "response": response_text,
                "model": model,
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return {"success": False, "error": str(e), "agent_id": self.agent_id}

    async def sketch_transpile(
        self, sketch_file_path: str, output_dir: str = "./swift-output", verbose: bool = False, save_files: bool = True
    ) -> Dict[str, Any]:
        """Convert Sketch design files to SwiftUI code using sketch-transpiler

        Args:
            sketch_file_path: Path to the .sketch file to transpile
            output_dir: Directory to save generated Swift files
            verbose: Enable verbose output
            save_files: Whether to save files to disk or just return code

        Returns:
            Dict containing Swift code files and generation status
        """
        try:
            import sys
            from pathlib import Path

            # Get the sketch-transpiler directory
            sketch_transpiler_path = Path(__file__).parent.parent.parent / "sketch-transpiler"

            if not sketch_transpiler_path.exists():
                return {"success": False, "error": f"sketch-transpiler directory not found at {sketch_transpiler_path}"}

            # Add sketch-transpiler src to Python path
            transpiler_src = sketch_transpiler_path / "src"
            if str(transpiler_src) not in sys.path:
                sys.path.insert(0, str(transpiler_src))

            # Also add the main sketch-transpiler path
            if str(sketch_transpiler_path) not in sys.path:
                sys.path.insert(0, str(sketch_transpiler_path))

            # Import sketch-transpiler components
            from src.sketch_parser import SketchParser
            from src.sketch_reader import SketchReader, SketchReaderError
            from src.swift_generator import SwiftGenerator

            # Validate input file
            sketch_file = Path(sketch_file_path)
            if not sketch_file.exists():
                return {"success": False, "error": f"Sketch file not found: {sketch_file_path}"}

            if not sketch_file.is_file():
                return {"success": False, "error": f"Path is not a file: {sketch_file_path}"}

            if sketch_file.suffix.lower() != '.sketch':
                return {"success": False, "error": f"File does not have .sketch extension: {sketch_file_path}"}

            # Create output directory if saving files
            output_path = Path(output_dir)
            if save_files:
                output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Agent {self.agent_id} transpiling Sketch file: {sketch_file_path}")

            # Step 1: Read and validate sketch file
            with SketchReader(str(sketch_file)) as reader:
                if not reader.validate_sketch_format():
                    return {"success": False, "error": "Invalid Sketch file format"}

                version_info = reader.get_version_info()

                # Step 2: Parse document structure
                parser = SketchParser(reader)
                document = parser.parse_document()

                total_artboards = len(document.get_all_artboards())
                if total_artboards == 0:
                    return {"success": False, "error": "No artboards or views found in the Sketch file"}

                # Step 3: Generate Swift code
                generator = SwiftGenerator(document)

                if save_files:
                    # Save files to disk
                    swift_files = generator.generate_all_views(output_path)

                    # Get file sizes for summary
                    file_summary = {}
                    for filename in swift_files.keys():
                        file_path = output_path / filename
                        if file_path.exists():
                            file_summary[filename] = {"size": file_path.stat().st_size, "path": str(file_path)}

                    return {
                        "success": True,
                        "sketch_file": str(sketch_file),
                        "output_directory": str(output_path),
                        "sketch_version": version_info.get("sketch_version", "unknown"),
                        "created_with": version_info.get("created_with", "unknown"),
                        "pages_count": len(document.pages),
                        "artboards_count": total_artboards,
                        "generated_files_count": len(swift_files),
                        "files": file_summary,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    # Return Swift code without saving to disk
                    swift_files = generator.generate_all_views()  # Without output_dir, returns code only

                    return {
                        "success": True,
                        "sketch_file": str(sketch_file),
                        "sketch_version": version_info.get("sketch_version", "unknown"),
                        "created_with": version_info.get("created_with", "unknown"),
                        "pages_count": len(document.pages),
                        "artboards_count": total_artboards,
                        "generated_files_count": len(swift_files),
                        "swift_code": swift_files,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    }

        except SketchReaderError as e:
            logger.error(f"Sketch file error in agent {self.agent_id}: {e}")
            return {"success": False, "error": f"Sketch file error: {e}", "agent_id": self.agent_id}
        except ImportError as e:
            logger.error(f"Failed to import sketch-transpiler in agent {self.agent_id}: {e}")
            return {"success": False, "error": f"sketch-transpiler not available: {e}", "agent_id": self.agent_id}
        except Exception as e:
            logger.error(f"Sketch transpilation failed in agent {self.agent_id}: {e}")
            return {"success": False, "error": str(e), "agent_id": self.agent_id}
