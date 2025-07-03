"""
Vector Memory Manager using ChromaDB

This module provides a vector-based memory system that replaces the JSON file-based
memory with semantic search capabilities using ChromaDB and sentence transformers.

Key Features:
- Semantic search through agent memories
- Context-aware memory retrieval
- Automatic embedding generation
- Efficient vector storage and retrieval
- Migration from JSON memory format
"""

from __future__ import annotations
import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from pathlib import Path

from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(
        "Sentence Transformers not installed. Install with: pip install sentence-transformers"
    )

# Import shared types
from reactive_agents.core.types.memory_types import AgentMemory
from reactive_agents.core.types.status_types import TaskStatus
from reactive_agents.core.types.session_types import AgentSession

if TYPE_CHECKING:
    from reactive_agents.core.context.agent_context import AgentContext
    from reactive_agents.utils.logging import Logger


class VectorMemoryConfig(BaseModel):
    """Configuration for vector memory system"""

    collection_name: str
    persist_directory: str = "storage/memory/vector"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_memory_items: int = 1000
    similarity_threshold: float = 0.7
    context_window_size: int = 5
    enable_metadata_filtering: bool = True
    auto_cleanup_enabled: bool = True
    cleanup_interval_days: int = 30

    def __init__(self, **data):
        # Sanitize collection name if provided
        if "collection_name" in data:
            data["collection_name"] = self._sanitize_collection_name(
                data["collection_name"]
            )
        super().__init__(**data)

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        """Sanitize collection name for ChromaDB compatibility."""
        # Replace spaces and invalid characters with underscores
        sanitized = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Remove any other invalid characters, keep only alphanumeric, dots, underscores, and hyphens
        import re

        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", sanitized)
        # Ensure it starts and ends with alphanumeric
        sanitized = sanitized.strip("._-")
        # Ensure minimum length
        if len(sanitized) < 3:
            sanitized = f"collection_{sanitized}"
        # Ensure maximum length
        if len(sanitized) > 512:
            sanitized = sanitized[:512]
        return sanitized


class MemoryItem(BaseModel):
    """Represents a single memory item in the vector store"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: str  # "session", "reflection", "tool_result", "context"
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None


class VectorMemoryManager(BaseModel):
    """
    ChromaDB-based vector memory manager for semantic search and retrieval.

    This replaces the JSON-based MemoryManager with a vector database that enables
    semantic search through agent memories, context-aware retrieval, and efficient
    storage of large memory collections.
    """

    context: Any = Field(exclude=True)  # Reference back to the main context
    config: VectorMemoryConfig

    # State
    agent_memory: Optional[AgentMemory] = None
    collection: Optional[Any] = None  # ChromaDB collection
    embedding_model: Optional[Any] = None  # SentenceTransformer model
    memory_enabled: bool = True
    _client: Optional[Any] = None  # ChromaDB client
    _ready: bool = False  # Flag to indicate if vector memory is ready

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        # Set default config if not provided
        if "config" not in data:
            data["config"] = VectorMemoryConfig(
                collection_name=data["context"].agent_name.replace(" ", "_").lower()
            )

        super().__init__(**data)
        self.memory_enabled = self.context.use_memory_enabled

        # Always initialize agent_memory immediately for traditional memory support
        self.agent_memory = AgentMemory(agent_name=self.context.agent_name)

        # Add initialization future for proper async coordination
        self._init_future = None

        if self.memory_enabled:
            # Initialize synchronously instead of asynchronously
            try:
                # Use asyncio.run if we're not already in an event loop
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, create a task and store the future
                    self._init_future = loop.create_task(
                        self._initialize_vector_memory()
                    )
                except RuntimeError:
                    # No event loop running, run synchronously
                    asyncio.run(self._initialize_vector_memory())
            except Exception as e:
                self.agent_logger.error(f"Failed to initialize vector memory: {e}")
                # Keep agent_memory initialized, just disable vector memory
                self.memory_enabled = False

    @property
    def agent_logger(self) -> Logger:
        assert self.context.agent_logger is not None
        return self.context.agent_logger

    async def _initialize_vector_memory(self):
        """Initialize the vector memory system with ChromaDB and embedding model."""

        if not self.memory_enabled:
            self.agent_logger.debug(
                "Vector memory is disabled, skipping initialization."
            )
            return

        try:
            # Check dependencies
            if not CHROMADB_AVAILABLE:
                self.agent_logger.warning(
                    "ChromaDB not available. Install with: pip install chromadb. Falling back to basic memory."
                )
                self.agent_memory = AgentMemory(agent_name=self.context.agent_name)
                self.memory_enabled = False
                return

            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                self.agent_logger.warning(
                    "Sentence Transformers not available. Install with: pip install sentence-transformers. Falling back to basic memory."
                )
                self.agent_memory = AgentMemory(agent_name=self.context.agent_name)
                self.memory_enabled = False
                return

            # Initialize ChromaDB client
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )

            # Get or create collection
            try:
                # First try to get the collection
                self.collection = self._client.get_collection(
                    name=self.config.collection_name
                )
                self.agent_logger.info(
                    f"Loaded existing vector memory collection: {self.config.collection_name}"
                )
            except Exception as e:
                # Collection doesn't exist or other error, create it
                try:
                    self.collection = self._client.create_collection(
                        name=self.config.collection_name,
                        metadata={"agent_name": self.context.agent_name},
                    )
                    self.agent_logger.info(
                        f"Created new vector memory collection: {self.config.collection_name}"
                    )
                except Exception as create_error:
                    self.agent_logger.error(
                        f"Failed to create collection: {create_error}"
                    )
                    raise create_error

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.agent_logger.info(
                f"Initialized embedding model: {self.config.embedding_model}"
            )

            # Migrate from JSON memory if it exists
            await self._migrate_from_json_memory()

            self.agent_logger.info("Vector memory system initialized successfully")
            self._ready = True

        except Exception as e:
            self.agent_logger.error(f"Failed to initialize vector memory: {e}")
            # Keep agent_memory initialized, just disable vector memory
            self.memory_enabled = False
            self._ready = False

    async def _migrate_from_json_memory(self):
        """Migrate existing JSON memory to vector format."""

        try:
            # Look for existing JSON memory file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_dir = os.path.join(base_dir, "memory")
            safe_agent_name = self.context.agent_name.replace(" ", "_").replace(
                "/", "_"
            )
            json_memory_path = os.path.join(
                memory_dir, f"{safe_agent_name}_memory.json"
            )

            if os.path.exists(json_memory_path):
                self.agent_logger.info(
                    f"Found existing JSON memory, migrating: {json_memory_path}"
                )

                with open(json_memory_path, "r") as f:
                    memory_data = json.load(f)

                # Load the JSON memory
                json_memory = AgentMemory(**memory_data)

                # Migrate session history
                for session in json_memory.session_history:
                    session_content = f"Session: {session.get('task', '')} | Status: {session.get('status', '')} | Result: {session.get('final_result_summary', '')}"
                    await self._store_memory_item(
                        content=session_content,
                        memory_type="session",
                        metadata=session or {},
                    )

                # Migrate reflections
                for reflection in json_memory.reflections:
                    await self._store_memory_item(
                        content=reflection.get("content", str(reflection)),
                        memory_type="reflection",
                        metadata=reflection or {},
                    )

                # Copy tool preferences and other data
                if self.agent_memory and json_memory.tool_preferences:
                    self.agent_memory.tool_preferences = json_memory.tool_preferences
                if self.agent_memory and json_memory.user_preferences:
                    self.agent_memory.user_preferences = json_memory.user_preferences

                self.agent_logger.info(
                    f"Successfully migrated {len(json_memory.session_history)} sessions and {len(json_memory.reflections)} reflections"
                )

                # Optionally backup and remove JSON file
                backup_path = f"{json_memory_path}.backup"
                os.rename(json_memory_path, backup_path)
                self.agent_logger.info(f"JSON memory backed up to: {backup_path}")
            else:
                self.agent_logger.debug("No existing JSON memory found to migrate")

        except Exception as e:
            self.agent_logger.warning(f"Failed to migrate JSON memory: {e}")

    async def _store_memory_item(
        self, content: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory item in the vector database."""

        if not self.is_ready():
            self.agent_logger.warning("Vector memory not ready, cannot store memory")
            return ""

        try:
            # Create memory item
            memory_item = MemoryItem(
                content=content, memory_type=memory_type, metadata=metadata or {}
            )

            # Generate embedding
            assert self.embedding_model is not None
            embedding = self.embedding_model.encode(content).tolist()
            memory_item.embedding = embedding

            # Prepare metadata for ChromaDB (convert sets to lists)
            chroma_metadata = self._prepare_metadata_for_chroma(
                {
                    "memory_type": memory_type,
                    "timestamp": memory_item.timestamp.isoformat(),
                    "agent_name": self.context.agent_name,
                    **(metadata or {}),
                }
            )

            # Store in ChromaDB
            assert self.collection is not None
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata],
                ids=[memory_item.id],
            )

            self.agent_logger.debug(f"Stored memory item: {memory_item.id}")

            return memory_item.id

        except Exception as e:
            self.agent_logger.error(f"Failed to store memory item: {e}")
            return ""

    async def store_memory(
        self, content: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory item in the vector database.

        Args:
            content: The memory content to store
            memory_type: Type of memory (e.g., "session", "reflection", "preference", "context")
            metadata: Optional metadata for the memory

        Returns:
            Memory item ID if successful, empty string otherwise
        """
        # Check for duplicate content before storing
        if self.is_ready():
            try:
                # Search for exact content match
                existing_memories = await self.search_memory(content, n_results=1)
                if existing_memories and existing_memories[0]["content"] == content:
                    self.agent_logger.debug(
                        f"Duplicate content detected, skipping storage: {content[:50]}..."
                    )
                    return existing_memories[0]["id"]
            except Exception as e:
                self.agent_logger.debug(f"Error checking for duplicates: {e}")

        return await self._store_memory_item(content, memory_type, metadata)

    def _prepare_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB by converting unsupported types."""
        chroma_metadata = {}

        for key, value in metadata.items():
            if isinstance(value, (set, list)):
                # Convert sets and lists to strings
                chroma_metadata[key] = str(value)
            elif isinstance(value, dict):
                # Convert complex objects to strings
                chroma_metadata[key] = str(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # These types are supported by ChromaDB
                chroma_metadata[key] = value
            else:
                # Convert other types to strings
                chroma_metadata[key] = str(value)

        return chroma_metadata

    async def search_memory(
        self, query: str, n_results: int = 5, memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search through memories using semantic similarity.

        Args:
            query: The search query
            n_results: Maximum number of results to return
            memory_types: Optional filter by memory types

        Returns:
            List of relevant memory items with metadata
        """

        if not self.is_ready():
            self.agent_logger.warning("Vector memory not ready, cannot search")
            return []

        try:
            # Handle empty query - use collection.get() instead of query()
            if not query.strip():
                self.agent_logger.debug(
                    "Empty query detected, using collection.get() to retrieve all memories"
                )

                if not self.collection:
                    self.agent_logger.warning("Collection not available")
                    return []

                assert self.collection is not None

                # Prepare filter for memory types
                where_filter = None
                if memory_types:
                    where_filter = {"memory_type": {"$in": memory_types}}

                # Get all memories with optional memory type filter
                # Note: collection.get() doesn't accept n_results, we'll get all and limit later
                results = self.collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"],
                )

                # Format results for empty query
                memories = []
                if results["documents"]:
                    for i, doc in enumerate(results["documents"]):
                        memory = {
                            "id": results["ids"][i],
                            "content": doc,
                            "distance": None,  # No distance for get() operation
                            "relevance_score": 1.0,  # Perfect relevance for empty query
                            "metadata": (
                                results["metadatas"][i] if results["metadatas"] else {}
                            ),
                        }
                        memories.append(memory)

                    # Limit results to n_results
                    memories = memories[:n_results]

                self.agent_logger.debug(
                    f"Retrieved {len(memories)} memories with empty query"
                )
                return memories

            # Generate query embedding for non-empty queries
            if not self.embedding_model:
                self.agent_logger.warning("Embedding model not available")
                return []

            assert self.embedding_model is not None
            query_embedding = self.embedding_model.encode(query).tolist()

            # Prepare filter for memory types
            where_filter = None
            if memory_types:
                where_filter = {"memory_type": {"$in": memory_types}}

            # Search in ChromaDB
            if not self.collection:
                self.agent_logger.warning("Collection not available")
                return []

            assert self.collection is not None

            # Add include parameter to get documents, metadatas, and distances
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            # Debug: Print raw results to see what we're getting back
            self.agent_logger.debug(f"Raw ChromaDB query results: {results}")

            # Format results
            memories = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else 0
                    )
                    # ChromaDB returns cosine distance, which ranges from 0 to 2
                    # 0 = identical, 2 = completely opposite
                    # Convert to similarity score: 1 - (distance / 2)
                    relevance_score = 1.0 - (distance / 2.0)

                    memory = {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "distance": distance,
                        "relevance_score": relevance_score,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"] and results["metadatas"][0]
                            else {}
                        ),
                    }

                    # Filter by similarity threshold
                    if relevance_score >= self.config.similarity_threshold:
                        memories.append(memory)
                    else:
                        self.agent_logger.debug(
                            f"Memory filtered out due to low similarity: {relevance_score:.3f} < {self.config.similarity_threshold}"
                        )
            else:
                self.agent_logger.debug(f"No documents returned from ChromaDB query")
                if results.get("ids") and results["ids"][0]:
                    self.agent_logger.debug(
                        f"But we have {len(results['ids'][0])} IDs returned"
                    )
                if results.get("distances") and results["distances"][0]:
                    self.agent_logger.debug(
                        f"And we have {len(results['distances'][0])} distances returned"
                    )

            self.agent_logger.debug(
                f"Found {len(memories)} relevant memories for query: {query[:50]}..."
            )
            return memories

        except Exception as e:
            self.agent_logger.error(f"Failed to search memories: {e}")
            return []

    async def get_context_memories(
        self, task: str, max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current task context.

        Args:
            task: The current task description
            max_items: Maximum number of context memories to return

        Returns:
            List of contextually relevant memories
        """

        try:
            # Search for task-related memories
            task_memories = await self.search_memory(
                task, n_results=max_items // 2, memory_types=["session", "context"]
            )

            # Search for tool-related memories if task mentions tools
            tool_memories = []
            if any(
                keyword in task.lower()
                for keyword in ["search", "database", "file", "sql", "web"]
            ):
                tool_memories = await self.search_memory(
                    task, n_results=max_items // 4, memory_types=["tool_result"]
                )

            # Get recent reflections
            reflection_memories = await self.search_memory(
                task, n_results=max_items // 4, memory_types=["reflection"]
            )

            # Combine and deduplicate
            all_memories = task_memories + tool_memories + reflection_memories
            seen_ids = set()
            unique_memories = []

            for memory in all_memories:
                if memory["id"] not in seen_ids:
                    seen_ids.add(memory["id"])
                    unique_memories.append(memory)

                if len(unique_memories) >= max_items:
                    break

            # If no memories found through semantic search, fall back to all memories
            if not unique_memories:
                self.agent_logger.debug(
                    "No memories found through semantic search, falling back to all memories"
                )
                fallback_memories = await self.search_memory(
                    "",
                    n_results=max_items,
                    memory_types=["session", "context", "preference"],
                )
                unique_memories = fallback_memories[:max_items]

            # Sort by relevance score
            unique_memories.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            self.agent_logger.debug(
                f"Retrieved {len(unique_memories)} context memories for task"
            )
            return unique_memories

        except Exception as e:
            self.agent_logger.error(f"Failed to get context memories: {e}")
            return []

    async def store_session_memory(self, session_data: AgentSession):
        """Store session completion as a memory item."""

        if not self.memory_enabled:
            return

        try:
            # Create a comprehensive session summary
            session_summary = f"""
            Task: {session_data.initial_task}
            Status: {session_data.task_status}
            Iterations: {session_data.iterations}
            Tools Used: {', '.join(session_data.successful_tools)}
            Final Result: {str(session_data.final_answer)[:200] if session_data.final_answer else 'None'}
            """

            # Store as memory
            await self._store_memory_item(
                content=session_summary.strip(),
                memory_type="session",
                metadata={
                    "session_id": session_data.session_id,
                    "task": session_data.initial_task,
                    "status": str(session_data.task_status),
                    "iterations": session_data.iterations,
                    "tools_used": session_data.successful_tools,
                    "success": session_data.task_status
                    in [TaskStatus.COMPLETE, TaskStatus.RESCOPED_COMPLETE],
                },
            )

            # Update traditional agent memory for compatibility
            if self.agent_memory:
                self.agent_memory.session_history.append(
                    {
                        "session_id": session_data.session_id,
                        "timestamp": datetime.now().isoformat(),
                        "task": session_data.initial_task,
                        "status": str(session_data.task_status),
                        "success": session_data.task_status
                        in [TaskStatus.COMPLETE, TaskStatus.RESCOPED_COMPLETE],
                        "iterations": session_data.iterations,
                        "tools_used": session_data.successful_tools,
                    }
                )

        except Exception as e:
            self.agent_logger.error(f"Failed to store session memory: {e}")

    async def store_reflection_memory(
        self, reflection_content: str, context: Optional[Dict[str, Any]] = None
    ):
        """Store a reflection as a memory item."""

        if not self.memory_enabled:
            return

        try:
            await self._store_memory_item(
                content=reflection_content,
                memory_type="reflection",
                metadata=context or {},
            )

            # Update traditional agent memory for compatibility
            if self.agent_memory:
                self.agent_memory.reflections.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "content": reflection_content,
                        "context": context or {},
                    }
                )

        except Exception as e:
            self.agent_logger.error(f"Failed to store reflection memory: {e}")

    async def store_tool_result_memory(
        self, tool_name: str, tool_input: str, tool_output: str, success: bool
    ):
        """Store tool execution result as a memory item."""

        if not self.memory_enabled:
            return

        try:
            tool_summary = f"Tool: {tool_name}\nInput: {tool_input}\nOutput: {tool_output[:200]}..."

            await self._store_memory_item(
                content=tool_summary,
                memory_type="tool_result",
                metadata={
                    "tool_name": tool_name,
                    "input": tool_input,
                    "output": tool_output,
                    "success": success,
                },
            )

        except Exception as e:
            self.agent_logger.error(f"Failed to store tool result memory: {e}")

    async def cleanup_old_memories(self, days_old: Optional[int] = None):
        """Clean up old memories based on age."""

        if not self.collection:
            return

        try:
            days_old = days_old or self.config.cleanup_interval_days
            cutoff_date = datetime.now() - timedelta(days=days_old)

            # Query old memories
            results = self.collection.get(
                where={"timestamp": {"$lt": cutoff_date.isoformat()}}
            )

            if results["ids"]:
                # Delete old memories
                self.collection.delete(ids=results["ids"])
                self.agent_logger.info(f"Cleaned up {len(results['ids'])} old memories")

        except Exception as e:
            self.agent_logger.error(f"Failed to cleanup old memories: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector memory system."""

        if not self.is_ready():
            return {
                "error": "Vector memory not ready",
                "memory_type": "vector",
                "vector_memory_enabled": self.memory_enabled,
                "ready": False,
            }

        try:
            # Get collection info
            assert self.collection is not None
            collection_count = self.collection.count()

            # Get memory type distribution
            type_distribution = {}
            for memory_type in ["session", "reflection", "tool_result", "context"]:
                results = self.collection.get(where={"memory_type": memory_type})
                type_distribution[memory_type] = (
                    len(results["ids"]) if results["ids"] else 0
                )

            return {
                "total_memories": collection_count,
                "memory_types": type_distribution,
                "collection_name": self.config.collection_name,
                "embedding_model": self.config.embedding_model,
                "similarity_threshold": self.config.similarity_threshold,
                "vector_memory_enabled": True,
                "ready": True,
            }

        except Exception as e:
            self.agent_logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    # Compatibility methods for existing MemoryManager interface
    def save_memory(self):
        """Save traditional agent memory (reflections, session history, tool preferences) to JSON file."""
        if not self.memory_enabled or not self.agent_memory:
            self.agent_logger.debug(
                "Memory saving skipped (disabled or no memory object)."
            )
            return

        # Skip JSON saving if vector memory is enabled and ready
        # Vector memory can handle all the same data types plus semantic search
        if self.is_ready():
            self.agent_logger.debug(
                "Vector memory enabled and ready - skipping JSON memory save (data stored in vector format)"
            )
            return

        try:
            # Update last_updated timestamp
            self.agent_memory.last_updated = datetime.now()

            # Sync reflections from reflection manager if available
            if self.context.reflection_manager:
                self.agent_memory.reflections = (
                    self.context.reflection_manager.reflections
                )

            # Use settings system for path resolution
            from reactive_agents.config.settings import get_settings

            settings = get_settings()

            # Get memory directory and create agent-specific file path
            memory_dir = settings.get_memory_path()
            safe_agent_name = self.context.agent_name.replace(" ", "_").replace(
                "/", "_"
            )
            memory_file_path = memory_dir / f"{safe_agent_name}_memory.json"

            # Ensure directory exists
            memory_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to JSON file
            memory_dict = self.agent_memory.dict()
            memory_json = json.dumps(memory_dict, indent=2, default=str)

            with open(memory_file_path, "w") as f:
                f.write(memory_json)

            self.agent_logger.debug(
                f"💾 Saved traditional memory to {memory_file_path}"
            )
            self.agent_logger.debug(
                "Vector memory auto-saves to ChromaDB, no manual save needed"
            )

        except Exception as e:
            self.agent_logger.error(f"Error saving traditional memory: {e}")

    def update_session_history(self, session_data: AgentSession):
        """Compatibility method - redirects to store_session_memory."""
        asyncio.create_task(self.store_session_memory(session_data))

    def update_tool_preferences(
        self, tool_name: str, success: bool, feedback: Optional[str] = None
    ):
        """Update tool preferences (maintains compatibility with existing interface)."""
        if not self.memory_enabled or not self.agent_memory:
            return

        try:
            if tool_name not in self.agent_memory.tool_preferences:
                self.agent_memory.tool_preferences[tool_name] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "total_usage": 0,
                    "feedback": [],
                    "success_rate": 0.0,
                }

            prefs = self.agent_memory.tool_preferences[tool_name]
            prefs["total_usage"] += 1

            if success:
                prefs["success_count"] += 1
            else:
                prefs["failure_count"] += 1

            prefs["success_rate"] = (
                (prefs["success_count"] / prefs["total_usage"])
                if prefs["total_usage"] > 0
                else 0.0
            )

            if feedback:
                prefs["feedback"].append(
                    {"timestamp": datetime.now().isoformat(), "message": feedback}
                )
                # Limit feedback history
                if len(prefs["feedback"]) > 5:
                    prefs["feedback"] = prefs["feedback"][-5:]

        except Exception as e:
            self.agent_logger.error(
                f"Error updating tool preferences for '{tool_name}': {e}"
            )

    def get_reflections(self) -> List[Dict[str, Any]]:
        """Compatibility method - returns reflections from agent memory."""
        if self.agent_memory:
            return self.agent_memory.reflections
        return []

    def get_tool_preferences(self) -> Dict[str, Any]:
        """Compatibility method - returns tool preferences."""
        if self.agent_memory:
            return self.agent_memory.tool_preferences
        return {}

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Compatibility method - returns session history."""
        if self.agent_memory:
            return self.agent_memory.session_history
        return []

    def is_ready(self) -> bool:
        """Check if vector memory is ready for use."""
        return (
            self._ready
            and self.collection is not None
            and self.embedding_model is not None
        )

    async def await_ready(self, timeout: Optional[float] = 30.0) -> bool:
        """
        Wait for vector memory to be ready for use.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            True if vector memory is ready, False if timeout or error occurred
        """
        if not self.memory_enabled:
            self.agent_logger.debug(
                "Vector memory is disabled, not waiting for initialization"
            )
            return False

        if self.is_ready():
            self.agent_logger.debug("Vector memory already ready")
            return True

        if self._init_future is None:
            self.agent_logger.warning(
                "No initialization future found, vector memory may not be initializing"
            )
            return False

        try:
            self.agent_logger.info(
                "Waiting for vector memory initialization to complete..."
            )
            await asyncio.wait_for(self._init_future, timeout=timeout)

            if self.is_ready():
                self.agent_logger.info(
                    "Vector memory initialization completed successfully"
                )
                return True
            else:
                self.agent_logger.warning(
                    "Vector memory initialization completed but not ready"
                )
                return False

        except asyncio.TimeoutError:
            self.agent_logger.error(
                f"Vector memory initialization timed out after {timeout} seconds"
            )
            return False
        except Exception as e:
            self.agent_logger.error(f"Vector memory initialization failed: {e}")
            return False
