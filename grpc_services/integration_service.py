"""
gRPC Integration Service Implementation
"""

import asyncio
import logging
import os

# Import generated protobuf classes
import sys
import time
from concurrent import futures
from datetime import datetime
from typing import Any, Dict, List, Optional

import grpc

from generated import agent_pb2, agent_pb2_grpc

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'generated'))


class IntegrationGRPCServer(agent_pb2_grpc.IntegrationServiceServicer):
    """gRPC server implementation for integration services."""

    def __init__(self):
        self.logger = logging.getLogger("integration_service")
        self.integrations = {}
        self.integration_status = {}

        # Initialize integrations
        self._initialize_integrations()

        self.logger.info("Integration service initialized")

    def _initialize_integrations(self):
        """Initialize all available integrations."""
        integrations = {
            'slack': SlackIntegration(),
            'git': GitIntegration(),
            'aws': AWSIntegration(),
            'jenkins': JenkinsIntegration(),
            'youtrack': YouTrackIntegration(),
        }

        for name, integration in integrations.items():
            self.integrations[name] = integration
            self.integration_status[name] = {
                'status': 'healthy',
                'last_check': datetime.now(),
                'message': 'Integration initialized',
                'metrics': {},
            }
            self.logger.info(f"Initialized {name} integration")

    async def ExecuteSlackOperation(
        self, request: agent_pb2.SlackOperationRequest, context
    ) -> agent_pb2.IntegrationResponse:
        """Execute Slack operations."""
        operation = request.operation
        parameters = dict(request.parameters)

        self.logger.info(f"Executing Slack operation: {operation}")

        try:
            start_time = time.time()
            slack_integration = self.integrations['slack']
            result = await slack_integration.execute_operation(operation, parameters)
            execution_time = time.time() - start_time

            # Update metrics
            self.integration_status['slack']['metrics']['last_execution_time'] = str(execution_time)
            self.integration_status['slack']['last_check'] = datetime.now()

            return agent_pb2.IntegrationResponse(
                success=True,
                message=f"Slack operation '{operation}' completed successfully",
                result_data=result,
                timestamp=self._current_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"Slack operation failed: {e}")
            self.integration_status['slack']['status'] = 'degraded'
            self.integration_status['slack']['message'] = str(e)

            return agent_pb2.IntegrationResponse(
                success=False,
                message=f"Slack operation failed: {str(e)}",
                result_data={},
                timestamp=self._current_timestamp(),
            )

    async def ExecuteGitOperation(
        self, request: agent_pb2.GitOperationRequest, context
    ) -> agent_pb2.IntegrationResponse:
        """Execute Git operations."""
        operation = request.operation
        parameters = dict(request.parameters)

        self.logger.info(f"Executing Git operation: {operation}")

        try:
            start_time = time.time()
            git_integration = self.integrations['git']
            result = await git_integration.execute_operation(operation, parameters)
            execution_time = time.time() - start_time

            # Update metrics
            self.integration_status['git']['metrics']['last_execution_time'] = str(execution_time)
            self.integration_status['git']['last_check'] = datetime.now()

            return agent_pb2.IntegrationResponse(
                success=True,
                message=f"Git operation '{operation}' completed successfully",
                result_data=result,
                timestamp=self._current_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"Git operation failed: {e}")
            self.integration_status['git']['status'] = 'degraded'
            self.integration_status['git']['message'] = str(e)

            return agent_pb2.IntegrationResponse(
                success=False,
                message=f"Git operation failed: {str(e)}",
                result_data={},
                timestamp=self._current_timestamp(),
            )

    async def ExecuteAWSOperation(
        self, request: agent_pb2.AWSOperationRequest, context
    ) -> agent_pb2.IntegrationResponse:
        """Execute AWS operations."""
        service = request.service
        operation = request.operation
        parameters = dict(request.parameters)

        self.logger.info(f"Executing AWS {service} operation: {operation}")

        try:
            start_time = time.time()
            aws_integration = self.integrations['aws']
            result = await aws_integration.execute_operation(service, operation, parameters)
            execution_time = time.time() - start_time

            # Update metrics
            self.integration_status['aws']['metrics']['last_execution_time'] = str(execution_time)
            self.integration_status['aws']['last_check'] = datetime.now()

            return agent_pb2.IntegrationResponse(
                success=True,
                message=f"AWS {service} operation '{operation}' completed successfully",
                result_data=result,
                timestamp=self._current_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"AWS operation failed: {e}")
            self.integration_status['aws']['status'] = 'degraded'
            self.integration_status['aws']['message'] = str(e)

            return agent_pb2.IntegrationResponse(
                success=False,
                message=f"AWS operation failed: {str(e)}",
                result_data={},
                timestamp=self._current_timestamp(),
            )

    async def ExecuteJenkinsOperation(
        self, request: agent_pb2.JenkinsOperationRequest, context
    ) -> agent_pb2.IntegrationResponse:
        """Execute Jenkins operations."""
        operation = request.operation
        parameters = dict(request.parameters)

        self.logger.info(f"Executing Jenkins operation: {operation}")

        try:
            start_time = time.time()
            jenkins_integration = self.integrations['jenkins']
            result = await jenkins_integration.execute_operation(operation, parameters)
            execution_time = time.time() - start_time

            # Update metrics
            self.integration_status['jenkins']['metrics']['last_execution_time'] = str(execution_time)
            self.integration_status['jenkins']['last_check'] = datetime.now()

            return agent_pb2.IntegrationResponse(
                success=True,
                message=f"Jenkins operation '{operation}' completed successfully",
                result_data=result,
                timestamp=self._current_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"Jenkins operation failed: {e}")
            self.integration_status['jenkins']['status'] = 'degraded'
            self.integration_status['jenkins']['message'] = str(e)

            return agent_pb2.IntegrationResponse(
                success=False,
                message=f"Jenkins operation failed: {str(e)}",
                result_data={},
                timestamp=self._current_timestamp(),
            )

    async def ExecuteYouTrackOperation(
        self, request: agent_pb2.YouTrackOperationRequest, context
    ) -> agent_pb2.IntegrationResponse:
        """Execute YouTrack operations."""
        operation = request.operation
        parameters = dict(request.parameters)

        self.logger.info(f"Executing YouTrack operation: {operation}")

        try:
            start_time = time.time()
            youtrack_integration = self.integrations['youtrack']
            result = await youtrack_integration.execute_operation(operation, parameters)
            execution_time = time.time() - start_time

            # Update metrics
            self.integration_status['youtrack']['metrics']['last_execution_time'] = str(execution_time)
            self.integration_status['youtrack']['last_check'] = datetime.now()

            return agent_pb2.IntegrationResponse(
                success=True,
                message=f"YouTrack operation '{operation}' completed successfully",
                result_data=result,
                timestamp=self._current_timestamp(),
            )

        except Exception as e:
            self.logger.error(f"YouTrack operation failed: {e}")
            self.integration_status['youtrack']['status'] = 'degraded'
            self.integration_status['youtrack']['message'] = str(e)

            return agent_pb2.IntegrationResponse(
                success=False,
                message=f"YouTrack operation failed: {str(e)}",
                result_data={},
                timestamp=self._current_timestamp(),
            )

    async def GetIntegrationStatus(
        self, request: agent_pb2.IntegrationStatusRequest, context
    ) -> agent_pb2.IntegrationStatusResponse:
        """Get status of a specific integration."""
        integration_name = request.integration_name

        if integration_name not in self.integration_status:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Integration {integration_name} not found")
            return agent_pb2.IntegrationStatusResponse()

        status_info = self.integration_status[integration_name]

        return agent_pb2.IntegrationStatusResponse(
            integration_name=integration_name,
            status=status_info['status'],
            message=status_info['message'],
            last_check=self._datetime_to_timestamp(status_info['last_check']),
            metrics={k: str(v) for k, v in status_info['metrics'].items()},
        )

    async def TestIntegration(
        self, request: agent_pb2.TestIntegrationRequest, context
    ) -> agent_pb2.TestIntegrationResponse:
        """Test an integration connection."""
        integration_name = request.integration_name
        test_parameters = dict(request.test_parameters)

        if integration_name not in self.integrations:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Integration {integration_name} not found")
            return agent_pb2.TestIntegrationResponse()

        try:
            start_time = time.time()
            integration = self.integrations[integration_name]
            test_result = await integration.test_connection(test_parameters)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Update status
            if test_result:
                self.integration_status[integration_name]['status'] = 'healthy'
                self.integration_status[integration_name]['message'] = 'Test passed'
            else:
                self.integration_status[integration_name]['status'] = 'degraded'
                self.integration_status[integration_name]['message'] = 'Test failed'

            self.integration_status[integration_name]['last_check'] = datetime.now()

            return agent_pb2.TestIntegrationResponse(
                success=test_result,
                message="Test passed" if test_result else "Test failed",
                response_time_ms=response_time,
            )

        except Exception as e:
            self.logger.error(f"Integration test failed for {integration_name}: {e}")
            self.integration_status[integration_name]['status'] = 'down'
            self.integration_status[integration_name]['message'] = str(e)

            return agent_pb2.TestIntegrationResponse(
                success=False, message=f"Test failed: {str(e)}", response_time_ms=0.0
            )

    async def ListIntegrations(
        self, request: agent_pb2.ListIntegrationsRequest, context
    ) -> agent_pb2.ListIntegrationsResponse:
        """List all available integrations."""
        active_only = request.active_only

        integration_names = []
        statuses = {}

        for name, status_info in self.integration_status.items():
            if active_only and status_info['status'] == 'down':
                continue

            integration_names.append(name)
            statuses[name] = agent_pb2.IntegrationStatusResponse(
                integration_name=name,
                status=status_info['status'],
                message=status_info['message'],
                last_check=self._datetime_to_timestamp(status_info['last_check']),
                metrics={k: str(v) for k, v in status_info['metrics'].items()},
            )

        return agent_pb2.ListIntegrationsResponse(integration_names=integration_names, statuses=statuses)

    def _current_timestamp(self):
        """Get current timestamp in protobuf format."""
        from google.protobuf.timestamp_pb2 import Timestamp

        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        return timestamp

    def _datetime_to_timestamp(self, dt: datetime):
        """Convert datetime to protobuf timestamp."""
        from google.protobuf.timestamp_pb2 import Timestamp

        timestamp = Timestamp()
        timestamp.FromDatetime(dt)
        return timestamp


class BaseIntegration:
    """Base class for all integrations."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"integration.{name}")

    async def execute_operation(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute an operation. Override in subclasses."""
        raise NotImplementedError()

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test connection. Override in subclasses."""
        return True


class SlackIntegration(BaseIntegration):
    """Slack integration implementation."""

    def __init__(self):
        super().__init__("slack")

    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Execute Slack operations."""
        # Simulate Slack operations
        await asyncio.sleep(0.5)  # Simulate network delay

        if operation == "send_message":
            channel = parameters.get('channel', '#general')
            message = parameters.get('message', 'Hello from MCP!')
            self.logger.info(f"Sending message to {channel}: {message}")
            return {'status': 'sent', 'channel': channel, 'message_id': f"msg_{int(time.time())}"}

        elif operation == "create_channel":
            channel_name = parameters.get('name', 'new-channel')
            self.logger.info(f"Creating channel: {channel_name}")
            return {'status': 'created', 'channel_id': f"C{int(time.time())}", 'channel_name': channel_name}

        elif operation == "get_users":
            self.logger.info("Fetching user list")
            return {'status': 'success', 'user_count': '42', 'users': 'user1,user2,user3'}

        else:
            raise ValueError(f"Unknown Slack operation: {operation}")

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test Slack connection."""
        # Simulate connection test
        await asyncio.sleep(0.2)
        return True


class GitIntegration(BaseIntegration):
    """Git/GitHub integration implementation."""

    def __init__(self):
        super().__init__("git")

    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Execute Git operations."""
        await asyncio.sleep(1.0)  # Simulate processing time

        if operation == "create_pr":
            repo = parameters.get('repo', 'owner/repo')
            title = parameters.get('title', 'New PR')
            branch = parameters.get('branch', 'feature-branch')
            self.logger.info(f"Creating PR in {repo}: {title}")
            return {
                'status': 'created',
                'pr_number': str(int(time.time()) % 1000),
                'url': f"https://github.com/{repo}/pull/{int(time.time()) % 1000}",
            }

        elif operation == "create_issue":
            repo = parameters.get('repo', 'owner/repo')
            title = parameters.get('title', 'New Issue')
            self.logger.info(f"Creating issue in {repo}: {title}")
            return {
                'status': 'created',
                'issue_number': str(int(time.time()) % 1000),
                'url': f"https://github.com/{repo}/issues/{int(time.time()) % 1000}",
            }

        elif operation == "get_commits":
            repo = parameters.get('repo', 'owner/repo')
            self.logger.info(f"Fetching commits for {repo}")
            return {'status': 'success', 'commit_count': '15', 'latest_commit': f"abc123{int(time.time()) % 1000}"}

        else:
            raise ValueError(f"Unknown Git operation: {operation}")

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test Git connection."""
        await asyncio.sleep(0.3)
        return True


class AWSIntegration(BaseIntegration):
    """AWS integration implementation."""

    def __init__(self):
        super().__init__("aws")

    async def execute_operation(self, service: str, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Execute AWS operations."""
        await asyncio.sleep(1.5)  # Simulate AWS API delay

        if service == "ec2":
            return await self._handle_ec2_operation(operation, parameters)
        elif service == "s3":
            return await self._handle_s3_operation(operation, parameters)
        elif service == "lambda":
            return await self._handle_lambda_operation(operation, parameters)
        else:
            raise ValueError(f"Unknown AWS service: {service}")

    async def _handle_ec2_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Handle EC2 operations."""
        if operation == "list_instances":
            self.logger.info("Listing EC2 instances")
            return {'status': 'success', 'instance_count': '3', 'instances': 'i-1234567,i-2345678,i-3456789'}

        elif operation == "start_instance":
            instance_id = parameters.get('instance_id', 'i-1234567')
            self.logger.info(f"Starting EC2 instance: {instance_id}")
            return {'status': 'starting', 'instance_id': instance_id, 'state': 'pending'}

        elif operation == "stop_instance":
            instance_id = parameters.get('instance_id', 'i-1234567')
            self.logger.info(f"Stopping EC2 instance: {instance_id}")
            return {'status': 'stopping', 'instance_id': instance_id, 'state': 'stopping'}

        else:
            raise ValueError(f"Unknown EC2 operation: {operation}")

    async def _handle_s3_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Handle S3 operations."""
        if operation == "list_buckets":
            self.logger.info("Listing S3 buckets")
            return {'status': 'success', 'bucket_count': '5', 'buckets': 'bucket1,bucket2,bucket3,bucket4,bucket5'}

        elif operation == "upload_file":
            bucket = parameters.get('bucket', 'my-bucket')
            key = parameters.get('key', 'file.txt')
            self.logger.info(f"Uploading file to S3: {bucket}/{key}")
            return {'status': 'uploaded', 'bucket': bucket, 'key': key, 'etag': f"etag_{int(time.time())}"}

        else:
            raise ValueError(f"Unknown S3 operation: {operation}")

    async def _handle_lambda_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Handle Lambda operations."""
        if operation == "invoke_function":
            function_name = parameters.get('function_name', 'my-function')
            self.logger.info(f"Invoking Lambda function: {function_name}")
            return {'status': 'success', 'function_name': function_name, 'execution_time': '1250', 'status_code': '200'}

        elif operation == "list_functions":
            self.logger.info("Listing Lambda functions")
            return {
                'status': 'success',
                'function_count': '8',
                'functions': 'func1,func2,func3,func4,func5,func6,func7,func8',
            }

        else:
            raise ValueError(f"Unknown Lambda operation: {operation}")

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test AWS connection."""
        await asyncio.sleep(0.5)
        return True


class JenkinsIntegration(BaseIntegration):
    """Jenkins integration implementation."""

    def __init__(self):
        super().__init__("jenkins")

    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Execute Jenkins operations."""
        await asyncio.sleep(2.0)  # Simulate Jenkins processing time

        if operation == "trigger_build":
            job_name = parameters.get('job_name', 'my-job')
            self.logger.info(f"Triggering Jenkins build: {job_name}")
            return {
                'status': 'triggered',
                'job_name': job_name,
                'build_number': str(int(time.time()) % 1000),
                'queue_id': str(int(time.time())),
            }

        elif operation == "get_build_status":
            job_name = parameters.get('job_name', 'my-job')
            build_number = parameters.get('build_number', '1')
            self.logger.info(f"Getting build status: {job_name}#{build_number}")
            return {
                'status': 'success',
                'job_name': job_name,
                'build_number': build_number,
                'build_status': 'SUCCESS',
                'duration': '120000',
            }

        elif operation == "list_jobs":
            self.logger.info("Listing Jenkins jobs")
            return {'status': 'success', 'job_count': '12', 'jobs': 'job1,job2,job3,job4,job5,job6'}

        else:
            raise ValueError(f"Unknown Jenkins operation: {operation}")

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test Jenkins connection."""
        await asyncio.sleep(0.4)
        return True


class YouTrackIntegration(BaseIntegration):
    """YouTrack integration implementation."""

    def __init__(self):
        super().__init__("youtrack")

    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, str]:
        """Execute YouTrack operations."""
        await asyncio.sleep(1.0)  # Simulate YouTrack API delay

        if operation == "create_issue":
            project = parameters.get('project', 'MY')
            summary = parameters.get('summary', 'New Issue')
            self.logger.info(f"Creating YouTrack issue in {project}: {summary}")
            return {
                'status': 'created',
                'issue_id': f"{project}-{int(time.time()) % 1000}",
                'url': f"https://youtrack.company.com/issue/{project}-{int(time.time()) % 1000}",
            }

        elif operation == "update_issue":
            issue_id = parameters.get('issue_id', 'MY-123')
            self.logger.info(f"Updating YouTrack issue: {issue_id}")
            return {'status': 'updated', 'issue_id': issue_id, 'updated_fields': 'state,assignee'}

        elif operation == "get_issues":
            project = parameters.get('project', 'MY')
            self.logger.info(f"Fetching issues for project: {project}")
            return {
                'status': 'success',
                'project': project,
                'issue_count': '25',
                'issues': f"{project}-1,{project}-2,{project}-3",
            }

        else:
            raise ValueError(f"Unknown YouTrack operation: {operation}")

    async def test_connection(self, parameters: Dict[str, Any]) -> bool:
        """Test YouTrack connection."""
        await asyncio.sleep(0.3)
        return True


async def start_integration_grpc_server(port: int = 50200):
    """Start gRPC server for integration services."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create server
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=15))

    # Create and add integration service
    integration_service = IntegrationGRPCServer()
    agent_pb2_grpc.add_IntegrationServiceServicer_to_server(integration_service, server)

    # Configure server address
    listen_addr = f'127.0.0.1:{port}'
    server.add_insecure_port(listen_addr)

    print(f"Starting Integration gRPC server on {listen_addr}")

    try:
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down Integration server...")
        await server.stop(5)


def run_integration_process(port: int = 50200):
    """Run integration service in separate process."""
    asyncio.run(start_integration_grpc_server(port))
