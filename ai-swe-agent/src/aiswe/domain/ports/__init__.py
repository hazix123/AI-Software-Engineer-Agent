"""Domain ports — the interfaces the outer layers must satisfy.

This package is the contract surface of Clean Architecture in this codebase.
Adding a capability means adding a Protocol/ABC here first, then implementing
it in `infrastructure/`. Nothing here may import `infrastructure`, `api`,
`agents`, or any third-party SDK (Pydantic excepted: used purely for
declarative validation, it carries no I/O).
"""

from aiswe.domain.ports.agent import (
    Agent,
    AgentDescriptor,
    AgentOutput,
    AgentRegistry,
)
from aiswe.domain.ports.cache import Cache, HealthCheck, LockManager
from aiswe.domain.ports.clock import Clock, SystemClock
from aiswe.domain.ports.event_bus import EventBus
from aiswe.domain.ports.llm import (
    Completion,
    EmbeddingProvider,
    LLMProvider,
    Message,
    StructuredCompletion,
    Usage,
)
from aiswe.domain.ports.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryStore,
    MemoryType,
    ScratchpadStore,
)
from aiswe.domain.ports.task_queue import Task, TaskQueue
from aiswe.domain.ports.tool import (
    CostClass,
    Permission,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from aiswe.domain.ports.unit_of_work import AbstractUnitOfWork, Repository

__all__ = [
    "AbstractUnitOfWork",
    "Agent",
    "AgentDescriptor",
    "AgentOutput",
    "AgentRegistry",
    "Cache",
    "Clock",
    "Completion",
    "CostClass",
    "EmbeddingProvider",
    "EventBus",
    "HealthCheck",
    "LLMProvider",
    "LockManager",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStore",
    "MemoryType",
    "Message",
    "Permission",
    "Repository",
    "ScratchpadStore",
    "StructuredCompletion",
    "SystemClock",
    "Task",
    "TaskQueue",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "Usage",
]
