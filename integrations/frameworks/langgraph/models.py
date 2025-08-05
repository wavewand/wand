"""
LangGraph Data Models

Defines data structures for LangGraph workflows, nodes, edges,
and stateful agent interactions.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from integrations.frameworks.base import BaseConfig, BaseQuery, BaseResponse


class LangGraphNodeType(str, Enum):
    """Types of LangGraph nodes."""

    FUNCTION = "function"
    CONDITIONAL = "conditional"
    TOOL = "tool"
    LLM = "llm"
    HUMAN = "human"
    START = "start"
    END = "end"
    ROUTER = "router"
    AGENT = "agent"


class LangGraphEdgeType(str, Enum):
    """Types of LangGraph edges."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"
    START = "start"
    END = "end"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class LangGraphConfig(BaseConfig):
    """LangGraph specific configuration."""

    # Workflow Configuration
    max_iterations: int = 100
    timeout_seconds: int = 300
    enable_streaming: bool = True

    # State Management
    state_schema: Dict[str, Any] = field(default_factory=dict)
    persist_state: bool = True
    state_storage_path: Optional[str] = "./storage/langgraph"

    # Execution Configuration
    parallel_execution: bool = False
    max_concurrent_nodes: int = 5
    retry_failed_nodes: bool = True
    max_retries: int = 3

    # Memory Configuration
    memory_type: str = "in_memory"  # in_memory, redis, database
    memory_config: Dict[str, Any] = field(default_factory=dict)

    # Tool Configuration
    tool_timeout: int = 30
    tool_retry_count: int = 2

    # LLM Configuration for workflow nodes
    default_llm_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if not self.state_schema:
            self.state_schema = {"messages": "list", "current_step": "string", "metadata": "dict"}
        if not self.memory_config:
            self.memory_config = {}
        if not self.default_llm_config:
            self.default_llm_config = {"model": "gpt-3.5-turbo", "temperature": 0.7}


@dataclass
class LangGraphState:
    """LangGraph workflow state."""

    state_id: str
    workflow_id: str
    current_node: Optional[str] = None

    # State data
    data: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def update_state(self, updates: Dict[str, Any]):
        """Update state data."""
        self.data.update(updates)
        self.updated_at = datetime.now()

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        self.updated_at = datetime.now()

    def add_execution_step(self, node_name: str, result: Any, duration: float):
        """Add execution step to history."""
        step = {
            "node_name": node_name,
            "result": str(result)[:500],  # Truncate long results
            "duration_ms": duration * 1000,
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
        }
        self.execution_history.append(step)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state_id": self.state_id,
            "workflow_id": self.workflow_id,
            "current_node": self.current_node,
            "data": self.data,
            "messages": self.messages,
            "metadata": self.metadata,
            "execution_history": self.execution_history,
            "iteration_count": self.iteration_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class LangGraphNode:
    """LangGraph workflow node."""

    node_id: str
    name: str
    node_type: LangGraphNodeType

    # Node function/logic
    function: Optional[Callable] = None
    function_name: Optional[str] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Conditional logic (for conditional nodes)
    condition: Optional[Callable] = None
    condition_name: Optional[str] = None

    # Tool configuration (for tool nodes)
    tool_name: Optional[str] = None
    tool_config: Dict[str, Any] = field(default_factory=dict)

    # Human interaction (for human nodes)
    human_prompt: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "function_name": self.function_name,
            "config": self.config,
            "condition_name": self.condition_name,
            "tool_name": self.tool_name,
            "tool_config": self.tool_config,
            "human_prompt": self.human_prompt,
            "input_schema": self.input_schema,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class LangGraphEdge:
    """LangGraph workflow edge."""

    edge_id: str
    from_node: str
    to_node: str
    edge_type: LangGraphEdgeType = LangGraphEdgeType.NORMAL

    # Conditional edge logic
    condition: Optional[Callable] = None
    condition_name: Optional[str] = None
    condition_map: Dict[str, str] = field(default_factory=dict)  # condition_result -> node_id

    # Edge metadata
    label: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type.value,
            "condition_name": self.condition_name,
            "condition_map": self.condition_map,
            "label": self.label,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class LangGraphWorkflow:
    """LangGraph workflow definition."""

    workflow_id: str
    name: str
    description: Optional[str] = None

    # Workflow structure
    nodes: Dict[str, LangGraphNode] = field(default_factory=dict)
    edges: Dict[str, LangGraphEdge] = field(default_factory=dict)

    # Workflow configuration
    start_node: Optional[str] = None
    end_nodes: List[str] = field(default_factory=list)

    # State configuration
    state_schema: Dict[str, Any] = field(default_factory=dict)

    # Execution configuration
    config: LangGraphConfig = field(default_factory=LangGraphConfig)

    # Metadata
    version: str = "1.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_node(self, node: LangGraphNode):
        """Add node to workflow."""
        self.nodes[node.node_id] = node
        self.updated_at = datetime.now()

    def add_edge(self, edge: LangGraphEdge):
        """Add edge to workflow."""
        self.edges[edge.edge_id] = edge
        self.updated_at = datetime.now()

    def set_start_node(self, node_id: str):
        """Set the starting node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in workflow")
        self.start_node = node_id
        self.updated_at = datetime.now()

    def add_end_node(self, node_id: str):
        """Add an end node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in workflow")
        if node_id not in self.end_nodes:
            self.end_nodes.append(node_id)
        self.updated_at = datetime.now()

    def validate(self) -> List[str]:
        """Validate workflow structure."""
        errors = []

        # Check for start node
        if not self.start_node:
            errors.append("No start node defined")
        elif self.start_node not in self.nodes:
            errors.append(f"Start node {self.start_node} not found")

        # Check for end nodes
        if not self.end_nodes:
            errors.append("No end nodes defined")

        # Check edges reference valid nodes
        for edge in self.edges.values():
            if edge.from_node not in self.nodes:
                errors.append(f"Edge {edge.edge_id} references unknown from_node: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"Edge {edge.edge_id} references unknown to_node: {edge.to_node}")

        # Check for unreachable nodes
        reachable_nodes = set()
        if self.start_node:
            self._find_reachable_nodes(self.start_node, reachable_nodes)

        unreachable = set(self.nodes.keys()) - reachable_nodes
        if unreachable:
            errors.append(f"Unreachable nodes: {list(unreachable)}")

        return errors

    def _find_reachable_nodes(self, node_id: str, reachable: set):
        """Recursively find reachable nodes."""
        if node_id in reachable:
            return

        reachable.add(node_id)

        # Find outgoing edges
        for edge in self.edges.values():
            if edge.from_node == node_id:
                self._find_reachable_nodes(edge.to_node, reachable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "state_schema": self.state_schema,
            "config": self.config.to_dict(),
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class LangGraphQuery(BaseQuery):
    """LangGraph query/execution request."""

    workflow_id: str
    input_data: Dict[str, Any] = field(default_factory=dict)

    # Execution configuration
    max_iterations: Optional[int] = None
    timeout_seconds: Optional[int] = None
    streaming: bool = False

    # State management
    state_id: Optional[str] = None  # Resume from existing state
    checkpoint: bool = True  # Save state at each step

    # Execution control
    start_node: Optional[str] = None  # Override default start node
    stop_conditions: List[str] = field(default_factory=list)

    # Human interaction
    human_input: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        # Set query_text for base class compatibility
        if not hasattr(self, 'query_text') or not self.query_text:
            self.query_text = json.dumps(self.input_data)


@dataclass
class LangGraphResponse(BaseResponse):
    """LangGraph execution response."""

    workflow_id: str
    state_id: str

    # Execution results
    output_data: Dict[str, Any] = field(default_factory=dict)
    final_state: Optional[Dict[str, Any]] = None

    # Execution metadata
    status: WorkflowStatus = WorkflowStatus.COMPLETED
    iterations_executed: int = 0
    execution_time_ms: float = 0

    # Node execution details
    node_results: List[Dict[str, Any]] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)

    # Human interaction
    waiting_for_human: bool = False
    human_prompt: Optional[str] = None

    # Streaming support
    is_streaming: bool = False
    stream_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # Set content for base class compatibility
        if not self.content and self.output_data:
            self.content = json.dumps(self.output_data, indent=2)

    def add_node_result(self, node_id: str, result: Any, duration_ms: float):
        """Add node execution result."""
        node_result = {
            "node_id": node_id,
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        }
        self.node_results.append(node_result)
        self.execution_path.append(node_id)

    def add_stream_event(self, event_type: str, data: Any):
        """Add streaming event."""
        event = {"event_type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
        self.stream_events.append(event)

    @classmethod
    def from_workflow_execution(
        cls, workflow_id: str, state_id: str, execution_result: Dict[str, Any], query: LangGraphQuery
    ) -> 'LangGraphResponse':
        """Create from workflow execution result."""
        return cls(
            workflow_id=workflow_id,
            state_id=state_id,
            output_data=execution_result.get('output', {}),
            final_state=execution_result.get('final_state'),
            status=WorkflowStatus(execution_result.get('status', 'completed')),
            iterations_executed=execution_result.get('iterations', 0),
            execution_time_ms=execution_result.get('execution_time_ms', 0),
            node_results=execution_result.get('node_results', []),
            execution_path=execution_result.get('execution_path', []),
            success=execution_result.get('status') == 'completed',
            framework="langgraph",
        )
