"""Types for workflow and orchestration."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import uuid
import time


class WorkflowNodeType(Enum):
    """Types of workflow nodes."""
    
    AGENT = "agent"
    CONDITION = "condition"  
    PARALLEL = "parallel"
    MERGE = "merge"
    DELAY = "delay"


class WorkflowNodeStatus(Enum):
    """Status of a workflow node."""
    
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNode(BaseModel):
    """A node in the workflow graph."""
    
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: WorkflowNodeType
    agent_name: Optional[str] = None  # For agent nodes
    task_template: Optional[str] = None  # Task with placeholders
    condition: Optional[str] = None  # For condition nodes
    delay_seconds: Optional[float] = None  # For delay nodes
    
    # Status and results
    status: WorkflowNodeStatus = WorkflowNodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    outputs_to: List[str] = Field(default_factory=list)
    
    # Context sharing
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    context_mapping: Dict[str, str] = Field(default_factory=dict)  # key -> context_path


class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow."""
    
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Workflow nodes
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict)
    
    # Global configuration
    global_context: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    parallel_execution: bool = True
    
    # Entry and exit points
    entry_nodes: List[str] = Field(default_factory=list)
    exit_nodes: List[str] = Field(default_factory=list)


class WorkflowExecutionResult(BaseModel):
    """Result of a workflow execution."""
    
    workflow_id: str
    execution_id: Optional[str] = None
    status: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    execution_time: Optional[float] = None
    success: bool = False
    
    # Node tracking
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: List[str] = Field(default_factory=list)
    
    # Node results
    node_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Global context after execution
    final_context: Dict[str, Any] = Field(default_factory=dict)
    final_result: Dict[str, Any] = Field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None