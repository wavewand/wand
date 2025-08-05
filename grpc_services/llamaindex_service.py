"""
LlamaIndex gRPC Service Implementation

This module implements the gRPC service for LlamaIndex AI operations including
RAG, document search, ingestion, and index management.
"""

import asyncio
import logging
import os

# Generated gRPC imports
import sys
from concurrent import futures
from typing import Any, Dict, Optional

import grpc
from grpc import aio

from ai_framework_registry import ai_framework_registry

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'protos'))

try:
    import agent_pb2
    import agent_pb2_grpc

    GRPC_AVAILABLE = True
except ImportError:
    logging.warning("gRPC protobuf files not available. Run: python -m grpc_tools.protoc")
    GRPC_AVAILABLE = False
    # Create mock classes for development

    class agent_pb2:
        class HaystackResponse:
            pass

        class HaystackDocument:
            pass

    class agent_pb2_grpc:
        class IntegrationServiceServicer:
            pass


# Framework registry imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class LlamaIndexService(agent_pb2_grpc.IntegrationServiceServicer):
    """gRPC service implementation for LlamaIndex operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize framework through registry
        self.framework = ai_framework_registry.get_framework("llamaindex")
        if not self.framework:
            self.logger.error("LlamaIndex framework not available in registry")
            raise RuntimeError("LlamaIndex framework not available")

        self.logger.info("Initialized LlamaIndex gRPC service")

    async def ExecuteHaystackRAG(self, request, context):
        """Execute a RAG (Retrieval-Augmented Generation) query using LlamaIndex."""
        try:
            self.logger.info(f"Executing RAG query with LlamaIndex: {request.query[:100]}...")

            # Extract parameters from request
            query = request.query
            index_name = request.pipeline_id or "default"
            temperature = request.temperature if request.temperature > 0 else 0.7
            max_tokens = request.max_tokens if request.max_tokens > 0 else 500
            context_params = dict(request.context) if request.context else {}

            # Execute RAG query through framework
            result = await self.framework.execute_rag_query(
                query=query,
                index_name=index_name,
                temperature=temperature,
                max_tokens=max_tokens,
                context=context_params,
            )

            # Convert sources to HaystackDocument messages
            source_documents = []
            if result.get("sources"):
                for source in result["sources"]:
                    doc = agent_pb2.HaystackDocument(
                        id=source.get("id", ""),
                        content=source.get("content", ""),
                        metadata={k: str(v) for k, v in source.get("metadata", {}).items()},
                        relevance_score=float(source.get("score", 0.0)),
                    )
                    source_documents.append(doc)

            # Build response
            response = agent_pb2.HaystackResponse(
                success=result.get("success", False),
                message=result.get("message", ""),
                answer=result.get("answer", ""),
                sources=source_documents,
                metadata={
                    "framework": "llamaindex",
                    "index_name": index_name,
                    "query_length": str(len(query)),
                    "temperature": str(temperature),
                    "max_tokens": str(max_tokens),
                },
                execution_time_ms=result.get("execution_time_ms", 0),
            )

            return response

        except Exception as e:
            self.logger.error(f"RAG query failed with LlamaIndex: {e}")
            return agent_pb2.HaystackResponse(
                success=False,
                message=f"RAG query failed: {str(e)}",
                answer="",
                sources=[],
                metadata={"error": str(e), "framework": "llamaindex"},
                execution_time_ms=0,
            )

    async def ExecuteHaystackSearch(self, request, context):
        """Execute a document search query using LlamaIndex."""
        try:
            self.logger.info(f"Executing search query with LlamaIndex: {request.query[:100]}...")

            # Extract parameters
            query = request.query
            search_type = request.search_type or "semantic"
            max_results = request.max_results if request.max_results > 0 else 10
            filters = dict(request.filters) if request.filters else {}

            # Execute search through framework
            result = await self.framework.search_documents(
                query=query, search_type=search_type, max_results=max_results, filters=filters
            )

            # Convert documents to HaystackDocument messages
            document_messages = []
            if result.get("documents"):
                for doc in result["documents"]:
                    doc_msg = agent_pb2.HaystackDocument(
                        id=doc.get("id", ""),
                        filename=doc.get("filename", ""),
                        content_type=doc.get("content_type", "text/plain"),
                        content=doc.get("content", "")[:1000],  # Limit content size
                        metadata={k: str(v) for k, v in doc.get("metadata", {}).items()},
                        relevance_score=float(doc.get("similarity_score", 0.0)),
                    )
                    document_messages.append(doc_msg)

            response = agent_pb2.HaystackResponse(
                success=result.get("success", False),
                message=result.get("message", "Search completed"),
                answer=f"Found {len(document_messages)} documents",
                sources=document_messages,
                metadata={
                    "framework": "llamaindex",
                    "search_type": search_type,
                    "total_results": str(result.get("total_results", 0)),
                    "query_length": str(len(query)),
                },
                execution_time_ms=0,
            )

            return response

        except Exception as e:
            self.logger.error(f"Search query failed with LlamaIndex: {e}")
            return agent_pb2.HaystackResponse(
                success=False,
                message=f"Search query failed: {str(e)}",
                answer="",
                sources=[],
                metadata={"error": str(e), "framework": "llamaindex"},
                execution_time_ms=0,
            )

    async def IngestHaystackDocument(self, request, context):
        """Ingest a document into the LlamaIndex document store."""
        try:
            self.logger.info(f"Ingesting document with LlamaIndex: {request.filename}")

            # Extract document information
            filename = request.filename
            content_type = request.content_type
            content = request.content
            metadata = dict(request.metadata) if request.metadata else {}
            index_name = request.pipeline_id or "default"

            # Ingest document through framework
            ingest_result = await self.framework.ingest_document(
                filename=filename, content=content, content_type=content_type, metadata=metadata, index_name=index_name
            )

            if ingest_result.get("success"):
                # Create response document
                response_doc = agent_pb2.HaystackDocument(
                    id=ingest_result["document_id"],
                    filename=filename,
                    content_type=content_type,
                    content="",  # Don't include full content in response
                    metadata={k: str(v) for k, v in ingest_result.get("metadata", {}).items()},
                )

                response = agent_pb2.HaystackResponse(
                    success=True,
                    message=ingest_result.get("message", "Document ingested successfully"),
                    answer=f"Document {filename} ingested with ID: {ingest_result['document_id']}",
                    sources=[response_doc],
                    metadata={
                        "framework": "llamaindex",
                        "document_id": ingest_result["document_id"],
                        "index_name": index_name,
                        "content_length": str(ingest_result.get("content_length", 0)),
                    },
                    execution_time_ms=0,
                )
            else:
                response = agent_pb2.HaystackResponse(
                    success=False,
                    message=ingest_result.get("message", "Document ingestion failed"),
                    answer="",
                    sources=[],
                    metadata={"error": ingest_result.get("message", "Unknown error"), "framework": "llamaindex"},
                    execution_time_ms=0,
                )

            return response

        except Exception as e:
            self.logger.error(f"Document ingestion failed with LlamaIndex: {e}")
            return agent_pb2.HaystackResponse(
                success=False,
                message=f"Document ingestion failed: {str(e)}",
                answer="",
                sources=[],
                metadata={"error": str(e), "framework": "llamaindex"},
                execution_time_ms=0,
            )

    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LlamaIndex service."""
        status = await self.framework.get_status()
        return {
            "service_name": "LlamaIndexService",
            "framework": "llamaindex",
            "framework_status": status,
            "capabilities": [
                "rag",
                "document_search",
                "question_answering",
                "document_processing",
                "index_management",
                "semantic_search",
                "summarization",
                "query_engines",
            ],
        }


async def serve_llamaindex_service(port: int = 50058):
    """Start the LlamaIndex gRPC service."""
    if not GRPC_AVAILABLE:
        logging.error("gRPC not available - cannot start LlamaIndex service")
        return

    server = aio.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add LlamaIndex service
    llamaindex_service = LlamaIndexService()
    agent_pb2_grpc.add_IntegrationServiceServicer_to_server(llamaindex_service, server)

    # Configure server
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)

    logging.info(f"Starting LlamaIndex gRPC service on {listen_addr}")

    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down LlamaIndex gRPC service...")
        await server.stop(5)


def main():
    """Main entry point for the LlamaIndex service."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    port = int(os.environ.get('LLAMAINDEX_SERVICE_PORT', 50058))

    try:
        asyncio.run(serve_llamaindex_service(port))
    except KeyboardInterrupt:
        logging.info("LlamaIndex service stopped")


if __name__ == "__main__":
    main()
