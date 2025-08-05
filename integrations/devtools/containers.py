"""
Container and infrastructure integrations for Wand
"""

import asyncio
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import yaml

from ..base.integration_base import BaseIntegration

logger = logging.getLogger(__name__)


class DockerIntegration(BaseIntegration):
    """Docker container management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "docker_host": os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
            "api_version": "1.41",
            "timeout": 30,
        }
        super().__init__("docker", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Docker integration"""
        try:
            # Test Docker availability
            result = await self._run_docker_command(["version", "--format", "json"])
            logger.info("✅ Docker integration initialized successfully")
        except Exception as e:
            logger.error(f"❌ Docker initialization failed: {e}")
            raise

    async def cleanup(self):
        """Cleanup Docker resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Docker health"""
        try:
            result = await self._run_docker_command(["version", "--format", "json"])
            if result.returncode == 0:
                import json

                version_info = json.loads(result.stdout)
                return {
                    "status": "healthy",
                    "version": version_info.get("Client", {}).get("Version", "unknown"),
                    "api_version": version_info.get("Client", {}).get("ApiVersion", "unknown"),
                }
            else:
                return {"status": "unhealthy", "error": "Docker not accessible"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Docker operations"""

        if operation == "list_containers":
            return await self._list_containers(**kwargs)
        elif operation == "start_container":
            return await self._start_container(**kwargs)
        elif operation == "stop_container":
            return await self._stop_container(**kwargs)
        elif operation == "build_image":
            return await self._build_image(**kwargs)
        elif operation == "pull_image":
            return await self._pull_image(**kwargs)
        elif operation == "run_container":
            return await self._run_container(**kwargs)
        elif operation == "get_logs":
            return await self._get_container_logs(**kwargs)
        elif operation == "exec_command":
            return await self._exec_command(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _list_containers(self, all_containers: bool = False) -> Dict[str, Any]:
        """List Docker containers"""
        try:
            cmd = ["ps", "--format", "json"]
            if all_containers:
                cmd.append("-a")

            result = await self._run_docker_command(cmd)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        import json

                        container = json.loads(line)
                        containers.append(
                            {
                                "id": container.get("ID", ""),
                                "name": container.get("Names", ""),
                                "image": container.get("Image", ""),
                                "status": container.get("Status", ""),
                                "ports": container.get("Ports", ""),
                                "created": container.get("CreatedAt", ""),
                            }
                        )

                return {"success": True, "containers": containers, "total": len(containers)}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_container(
        self,
        image: str,
        name: Optional[str] = None,
        ports: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        detach: bool = True,
    ) -> Dict[str, Any]:
        """Run Docker container"""
        try:
            cmd = ["run"]

            if detach:
                cmd.append("-d")

            if name:
                cmd.extend(["--name", name])

            if ports:
                for host_port, container_port in ports.items():
                    cmd.extend(["-p", f"{host_port}:{container_port}"])

            if environment:
                for key, value in environment.items():
                    cmd.extend(["-e", f"{key}={value}"])

            if volumes:
                for host_path, container_path in volumes.items():
                    cmd.extend(["-v", f"{host_path}:{container_path}"])

            cmd.append(image)

            result = await self._run_docker_command(cmd)
            if result.returncode == 0:
                container_id = result.stdout.strip()
                return {"success": True, "container_id": container_id, "image": image, "name": name, "detached": detach}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _build_image(self, dockerfile_path: str, tag: str, build_context: str = ".") -> Dict[str, Any]:
        """Build Docker image"""
        try:
            cmd = ["build", "-t", tag, "-f", dockerfile_path, build_context]

            result = await self._run_docker_command(cmd)
            if result.returncode == 0:
                return {"success": True, "tag": tag, "dockerfile": dockerfile_path, "context": build_context}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _get_container_logs(self, container_id: str, tail: int = 100) -> Dict[str, Any]:
        """Get container logs"""
        try:
            cmd = ["logs", "--tail", str(tail), container_id]

            result = await self._run_docker_command(cmd)
            if result.returncode == 0:
                return {"success": True, "container_id": container_id, "logs": result.stdout, "tail": tail}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_docker_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run Docker command asynchronously"""
        full_cmd = ["docker"] + cmd

        process = await asyncio.create_subprocess_exec(
            *full_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config["timeout"])

            return subprocess.CompletedProcess(full_cmd, process.returncode, stdout.decode(), stderr.decode())

        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Docker command timed out after {self.config['timeout']} seconds")


class KubernetesIntegration(BaseIntegration):
    """Kubernetes cluster management integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "kubeconfig_path": os.getenv("KUBECONFIG", "~/.kube/config"),
            "namespace": "default",
            "timeout": 60,
        }
        super().__init__("kubernetes", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Kubernetes integration"""
        try:
            # Test kubectl availability
            result = await self._run_kubectl_command(["version", "--client", "--output", "json"])
            logger.info("✅ Kubernetes integration initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Kubernetes initialization failed: {e}")

    async def cleanup(self):
        """Cleanup Kubernetes resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Kubernetes cluster health"""
        try:
            result = await self._run_kubectl_command(["cluster-info"])
            if result.returncode == 0:
                return {"status": "healthy", "cluster_info": result.stdout}
            else:
                return {"status": "unhealthy", "error": result.stderr}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Kubernetes operations"""

        if operation == "get_pods":
            return await self._get_pods(**kwargs)
        elif operation == "get_services":
            return await self._get_services(**kwargs)
        elif operation == "apply_manifest":
            return await self._apply_manifest(**kwargs)
        elif operation == "delete_resource":
            return await self._delete_resource(**kwargs)
        elif operation == "get_logs":
            return await self._get_pod_logs(**kwargs)
        elif operation == "scale_deployment":
            return await self._scale_deployment(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _get_pods(self, namespace: Optional[str] = None, selector: Optional[str] = None) -> Dict[str, Any]:
        """Get Kubernetes pods"""
        try:
            cmd = ["get", "pods", "-o", "json"]

            if namespace:
                cmd.extend(["-n", namespace])
            elif self.config["namespace"]:
                cmd.extend(["-n", self.config["namespace"]])

            if selector:
                cmd.extend(["-l", selector])

            result = await self._run_kubectl_command(cmd)
            if result.returncode == 0:
                import json

                pods_data = json.loads(result.stdout)

                pods = []
                for pod in pods_data.get("items", []):
                    pods.append(
                        {
                            "name": pod["metadata"]["name"],
                            "namespace": pod["metadata"]["namespace"],
                            "status": pod["status"]["phase"],
                            "ready": self._get_pod_ready_status(pod),
                            "restarts": sum(
                                container.get("restartCount", 0)
                                for container in pod["status"].get("containerStatuses", [])
                            ),
                            "age": pod["metadata"]["creationTimestamp"],
                        }
                    )

                return {"success": True, "pods": pods, "total": len(pods)}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _apply_manifest(self, manifest_path: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Apply Kubernetes manifest"""
        try:
            cmd = ["apply", "-f", manifest_path]

            if namespace:
                cmd.extend(["-n", namespace])

            result = await self._run_kubectl_command(cmd)
            if result.returncode == 0:
                return {"success": True, "manifest": manifest_path, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_pod_ready_status(self, pod: Dict) -> str:
        """Get pod ready status string"""
        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        if not container_statuses:
            return "0/0"

        ready_count = sum(1 for status in container_statuses if status.get("ready", False))
        total_count = len(container_statuses)
        return f"{ready_count}/{total_count}"

    async def _run_kubectl_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run kubectl command asynchronously"""
        full_cmd = ["kubectl"] + cmd

        # Set kubeconfig if specified
        env = os.environ.copy()
        if self.config["kubeconfig_path"]:
            env["KUBECONFIG"] = os.path.expanduser(self.config["kubeconfig_path"])

        process = await asyncio.create_subprocess_exec(
            *full_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config["timeout"])

            return subprocess.CompletedProcess(full_cmd, process.returncode, stdout.decode(), stderr.decode())

        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"kubectl command timed out after {self.config['timeout']} seconds")


class TerraformIntegration(BaseIntegration):
    """Terraform infrastructure as code integration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "working_directory": ".",
            "terraform_binary": "terraform",
            "auto_approve": False,
            "timeout": 300,  # 5 minutes
        }
        super().__init__("terraform", {**default_config, **(config or {})})

    async def initialize(self):
        """Initialize Terraform integration"""
        try:
            # Test Terraform availability
            result = await self._run_terraform_command(["version", "-json"])
            logger.info("✅ Terraform integration initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Terraform initialization failed: {e}")

    async def cleanup(self):
        """Cleanup Terraform resources"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check Terraform health"""
        try:
            result = await self._run_terraform_command(["version", "-json"])
            if result.returncode == 0:
                import json

                version_info = json.loads(result.stdout)
                return {"status": "healthy", "version": version_info.get("terraform_version", "unknown")}
            else:
                return {"status": "unhealthy", "error": result.stderr}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def _execute_operation_impl(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute Terraform operations"""

        if operation == "init":
            return await self._terraform_init(**kwargs)
        elif operation == "plan":
            return await self._terraform_plan(**kwargs)
        elif operation == "apply":
            return await self._terraform_apply(**kwargs)
        elif operation == "destroy":
            return await self._terraform_destroy(**kwargs)
        elif operation == "validate":
            return await self._terraform_validate(**kwargs)
        elif operation == "show":
            return await self._terraform_show(**kwargs)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    async def _terraform_init(self, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Initialize Terraform working directory"""
        try:
            cmd = ["init"]
            result = await self._run_terraform_command(cmd, working_dir)

            if result.returncode == 0:
                return {
                    "success": True,
                    "operation": "init",
                    "working_directory": working_dir or self.config["working_directory"],
                    "output": result.stdout,
                }
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _terraform_plan(
        self, var_file: Optional[str] = None, working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create Terraform execution plan"""
        try:
            cmd = ["plan", "-no-color"]

            if var_file:
                cmd.extend(["-var-file", var_file])

            result = await self._run_terraform_command(cmd, working_dir)

            if result.returncode == 0:
                return {
                    "success": True,
                    "operation": "plan",
                    "working_directory": working_dir or self.config["working_directory"],
                    "plan_output": result.stdout,
                    "var_file": var_file,
                }
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _terraform_apply(
        self, var_file: Optional[str] = None, working_dir: Optional[str] = None, auto_approve: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Apply Terraform configuration"""
        try:
            cmd = ["apply", "-no-color"]

            if auto_approve or self.config["auto_approve"]:
                cmd.append("-auto-approve")

            if var_file:
                cmd.extend(["-var-file", var_file])

            result = await self._run_terraform_command(cmd, working_dir)

            if result.returncode == 0:
                return {
                    "success": True,
                    "operation": "apply",
                    "working_directory": working_dir or self.config["working_directory"],
                    "output": result.stdout,
                    "var_file": var_file,
                    "auto_approved": auto_approve or self.config["auto_approve"],
                }
            else:
                return {"success": False, "error": result.stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_terraform_command(
        self, cmd: List[str], working_dir: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Run Terraform command asynchronously"""
        full_cmd = [self.config["terraform_binary"]] + cmd
        work_dir = working_dir or self.config["working_directory"]

        process = await asyncio.create_subprocess_exec(
            *full_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=work_dir
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config["timeout"])

            return subprocess.CompletedProcess(full_cmd, process.returncode, stdout.decode(), stderr.decode())

        except asyncio.TimeoutError:
            process.kill()
            raise Exception(f"Terraform command timed out after {self.config['timeout']} seconds")
