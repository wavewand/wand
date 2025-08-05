"""
Batch Processing Module

Provides batch processing capabilities for bulk operations across AI frameworks.
Supports batch document ingestion, bulk RAG queries, and parallel processing.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BatchStatus(str, Enum):
    """Status of batch operations."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchResult:
    """Result of a batch operation."""

    batch_id: str
    status: BatchStatus
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[Dict[str, Any]] = None
    errors: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "batch_id": self.batch_id,
            "status": self.status.value,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": (self.successful_items / self.total_items * 100) if self.total_items > 0 else 0,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_seconds": ((self.end_time or datetime.now()) - self.start_time).total_seconds(),
            "results": self.results or [],
            "errors": self.errors or [],
        }


class BatchProcessor:
    """Handles batch processing operations for AI frameworks."""

    def __init__(self, max_concurrent: int = 5, max_batch_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_batch_size = max_batch_size
        self.active_batches: Dict[str, BatchResult] = {}
        self.logger = logging.getLogger(__name__)

        # Semaphore to limit concurrent operations
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch_rag_queries(
        self, queries: List[str], framework: str, framework_obj: Any, batch_config: Dict[str, Any] = None
    ) -> BatchResult:
        """Process multiple RAG queries in batch."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or {}

        if len(queries) > self.max_batch_size:
            raise ValueError(f"Batch size {len(queries)} exceeds maximum {self.max_batch_size}")

        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            total_items=len(queries),
            processed_items=0,
            successful_items=0,
            failed_items=0,
            start_time=datetime.now(),
            results=[],
            errors=[],
        )

        self.active_batches[batch_id] = batch_result
        self.logger.info(f"Starting batch RAG processing: {batch_id} with {len(queries)} queries")

        try:
            # Process queries concurrently
            tasks = []
            for i, query in enumerate(queries):
                task = self._process_single_rag_query(framework_obj, query, i, batch_config)
                tasks.append(task)

            # Execute with concurrency limit
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                batch_result.processed_items += 1

                if isinstance(result, Exception):
                    batch_result.failed_items += 1
                    batch_result.errors.append({"query_index": i, "query": queries[i], "error": str(result)})
                elif result.get("success", False):
                    batch_result.successful_items += 1
                    batch_result.results.append({"query_index": i, "query": queries[i], "result": result})
                else:
                    batch_result.failed_items += 1
                    batch_result.errors.append(
                        {"query_index": i, "query": queries[i], "error": result.get("error", "Unknown error")}
                    )

            # Determine final status
            if batch_result.failed_items == 0:
                batch_result.status = BatchStatus.COMPLETED
            elif batch_result.successful_items == 0:
                batch_result.status = BatchStatus.FAILED
            else:
                batch_result.status = BatchStatus.PARTIAL

            batch_result.end_time = datetime.now()

            self.logger.info(
                f"Batch RAG processing completed: {batch_id} - "
                f"{batch_result.successful_items}/{batch_result.total_items} successful"
            )

            return batch_result

        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.end_time = datetime.now()
            batch_result.errors.append({"error": f"Batch processing failed: {str(e)}"})
            self.logger.error(f"Batch RAG processing failed: {batch_id} - {e}")
            return batch_result

    async def _process_single_rag_query(
        self, framework_obj: Any, query: str, index: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single RAG query with concurrency control."""
        async with self.semaphore:
            try:
                result = await framework_obj.execute_rag_query(query=query, **config)
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def process_batch_document_ingestion(
        self, documents: List[Dict[str, Any]], framework: str, framework_obj: Any, batch_config: Dict[str, Any] = None
    ) -> BatchResult:
        """Process multiple document ingestions in batch."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or {}

        if len(documents) > self.max_batch_size:
            raise ValueError(f"Batch size {len(documents)} exceeds maximum {self.max_batch_size}")

        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            total_items=len(documents),
            processed_items=0,
            successful_items=0,
            failed_items=0,
            start_time=datetime.now(),
            results=[],
            errors=[],
        )

        self.active_batches[batch_id] = batch_result
        self.logger.info(f"Starting batch document ingestion: {batch_id} with {len(documents)} documents")

        try:
            # Process documents concurrently
            tasks = []
            for i, doc in enumerate(documents):
                task = self._process_single_document_ingestion(framework_obj, doc, i, batch_config)
                tasks.append(task)

            # Execute with concurrency limit
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                doc = documents[i]
                batch_result.processed_items += 1

                if isinstance(result, Exception):
                    batch_result.failed_items += 1
                    batch_result.errors.append(
                        {"document_index": i, "filename": doc.get("filename", f"document_{i}"), "error": str(result)}
                    )
                elif result.get("success", False):
                    batch_result.successful_items += 1
                    batch_result.results.append(
                        {"document_index": i, "filename": doc.get("filename", f"document_{i}"), "result": result}
                    )
                else:
                    batch_result.failed_items += 1
                    batch_result.errors.append(
                        {
                            "document_index": i,
                            "filename": doc.get("filename", f"document_{i}"),
                            "error": result.get("error", "Unknown error"),
                        }
                    )

            # Determine final status
            if batch_result.failed_items == 0:
                batch_result.status = BatchStatus.COMPLETED
            elif batch_result.successful_items == 0:
                batch_result.status = BatchStatus.FAILED
            else:
                batch_result.status = BatchStatus.PARTIAL

            batch_result.end_time = datetime.now()

            self.logger.info(
                f"Batch document ingestion completed: {batch_id} - "
                f"{batch_result.successful_items}/{batch_result.total_items} successful"
            )

            return batch_result

        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.end_time = datetime.now()
            batch_result.errors.append({"error": f"Batch processing failed: {str(e)}"})
            self.logger.error(f"Batch document ingestion failed: {batch_id} - {e}")
            return batch_result

    async def _process_single_document_ingestion(
        self, framework_obj: Any, document: Dict[str, Any], index: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single document ingestion with concurrency control."""
        async with self.semaphore:
            try:
                result = await framework_obj.ingest_document(
                    filename=document.get("filename", f"document_{index}"),
                    content=document.get("content", ""),
                    content_type=document.get("content_type", "text/plain"),
                    metadata=document.get("metadata", {}),
                    **config,
                )
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def process_batch_search(
        self, queries: List[str], framework: str, framework_obj: Any, batch_config: Dict[str, Any] = None
    ) -> BatchResult:
        """Process multiple search queries in batch."""
        batch_id = str(uuid.uuid4())
        batch_config = batch_config or {}

        if len(queries) > self.max_batch_size:
            raise ValueError(f"Batch size {len(queries)} exceeds maximum {self.max_batch_size}")

        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            total_items=len(queries),
            processed_items=0,
            successful_items=0,
            failed_items=0,
            start_time=datetime.now(),
            results=[],
            errors=[],
        )

        self.active_batches[batch_id] = batch_result
        self.logger.info(f"Starting batch search processing: {batch_id} with {len(queries)} queries")

        try:
            # Process searches concurrently
            tasks = []
            for i, query in enumerate(queries):
                task = self._process_single_search_query(framework_obj, query, i, batch_config)
                tasks.append(task)

            # Execute with concurrency limit
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                batch_result.processed_items += 1

                if isinstance(result, Exception):
                    batch_result.failed_items += 1
                    batch_result.errors.append({"query_index": i, "query": queries[i], "error": str(result)})
                elif result.get("success", False):
                    batch_result.successful_items += 1
                    batch_result.results.append({"query_index": i, "query": queries[i], "result": result})
                else:
                    batch_result.failed_items += 1
                    batch_result.errors.append(
                        {"query_index": i, "query": queries[i], "error": result.get("error", "Unknown error")}
                    )

            # Determine final status
            if batch_result.failed_items == 0:
                batch_result.status = BatchStatus.COMPLETED
            elif batch_result.successful_items == 0:
                batch_result.status = BatchStatus.FAILED
            else:
                batch_result.status = BatchStatus.PARTIAL

            batch_result.end_time = datetime.now()

            self.logger.info(
                f"Batch search processing completed: {batch_id} - "
                f"{batch_result.successful_items}/{batch_result.total_items} successful"
            )

            return batch_result

        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.end_time = datetime.now()
            batch_result.errors.append({"error": f"Batch processing failed: {str(e)}"})
            self.logger.error(f"Batch search processing failed: {batch_id} - {e}")
            return batch_result

    async def _process_single_search_query(
        self, framework_obj: Any, query: str, index: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single search query with concurrency control."""
        async with self.semaphore:
            try:
                result = await framework_obj.search_documents(query=query, **config)
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a batch operation."""
        if batch_id in self.active_batches:
            return self.active_batches[batch_id].to_dict()
        return None

    def list_active_batches(self) -> List[Dict[str, Any]]:
        """List all active batch operations."""
        return [batch.to_dict() for batch in self.active_batches.values()]

    def cleanup_completed_batches(self, max_age_hours: int = 24):
        """Clean up completed batch operations older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for batch_id, batch_result in self.active_batches.items():
            if (
                batch_result.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.PARTIAL]
                and batch_result.end_time
                and batch_result.end_time < cutoff_time
            ):
                to_remove.append(batch_id)

        for batch_id in to_remove:
            del self.active_batches[batch_id]

        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old batch operations")


# Global batch processor instance
batch_processor = BatchProcessor()


# Pydantic models for API requests
class BatchRAGRequest(BaseModel):
    queries: List[str]
    framework: str = "haystack"
    framework_config: Optional[Dict[str, Any]] = None


class BatchDocumentRequest(BaseModel):
    documents: List[Dict[str, Any]]
    framework: str = "haystack"
    framework_config: Optional[Dict[str, Any]] = None


class BatchSearchRequest(BaseModel):
    queries: List[str]
    framework: str = "haystack"
    framework_config: Optional[Dict[str, Any]] = None
