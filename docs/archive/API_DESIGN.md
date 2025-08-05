# AI Framework-Agnostic REST API Design

## Overview

The REST API has been refactored to be framework-agnostic, allowing support for multiple AI frameworks like Haystack, LlamaIndex, LangChain, and custom implementations. The framework is specified in the request payload rather than being hardcoded into the endpoint paths.

## New API Endpoints

### 🤖 **RAG Operations**
```
POST /api/v1/rag
```
Execute Retrieval-Augmented Generation queries.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "framework": "haystack",
  "pipeline_id": "default_rag",
  "temperature": 0.7,
  "max_tokens": 500,
  "context": {
    "domain": "ai"
  },
  "framework_config": {
    "custom_option": "value"
  }
}
```

### 🔍 **Document Search**
```
POST /api/v1/search
```
Search documents across frameworks.

**Request Body:**
```json
{
  "query": "artificial intelligence",
  "framework": "haystack",
  "search_type": "semantic",
  "max_results": 10,
  "filters": {
    "content_type": "pdf"
  },
  "framework_config": {}
}
```

### 📄 **Document Management**
```
POST /api/v1/documents      # Ingest document
GET /api/v1/documents       # List documents
GET /api/v1/documents/{id}  # Get document
DELETE /api/v1/documents/{id} # Delete document
```

**Ingest Request:**
```json
{
  "filename": "document.pdf",
  "content": "Document content...",
  "framework": "haystack",
  "content_type": "application/pdf",
  "metadata": {
    "author": "John Doe",
    "category": "research"
  },
  "framework_config": {}
}
```

**List/Get Parameters:**
```
?framework=haystack&content_type=pdf&limit=50
```

### 📝 **Text Summarization**
```
POST /api/v1/summarize
```

**Request Body:**
```json
{
  "text": "Long text to summarize...",
  "framework": "haystack",
  "pipeline_id": "default_summarization",
  "max_length": 200,
  "min_length": 50,
  "framework_config": {}
}
```

### ⚙️ **Pipeline Management**
```
GET /api/v1/pipelines           # List pipelines
GET /api/v1/pipelines/{id}      # Get pipeline info
```

**Parameters:**
```
?framework=haystack
```

### 🧠 **AI Framework Management**
```
GET /api/v1/ai/frameworks       # List all frameworks
GET /api/v1/ai/status          # Get all framework status
```

## Framework Support

### ✅ **Supported Frameworks**
- **Haystack** (`framework: "haystack"`) - ✅ Fully implemented
- **LlamaIndex** (`framework: "llamaindex"`) - ✅ Fully implemented

### 🔄 **Framework Selection**
The framework is specified via the `framework` field in request payloads:
- Defaults to `"haystack"` if not specified
- Validates framework availability before processing
- Returns appropriate error messages for unsupported frameworks

## Request/Response Format

### **Common Request Fields**
All AI operation requests include:
- `framework`: Target AI framework (default: "haystack")
- `framework_config`: Framework-specific configuration options

### **Common Response Format**
```json
{
  "success": true/false,
  "framework": "haystack",
  "result": { /* operation-specific data */ },
  "error": "Error message if failed"
}
```

### **Error Responses**
- `503 Service Unavailable`: AI frameworks not available
- `400 Bad Request`: Unsupported framework specified
- `404 Not Found`: Resource not found (documents, pipelines)
- `500 Internal Server Error`: Processing failed

## WebSocket Events

Real-time events are broadcast for AI operations:
```json
{
  "type": "ai_event",
  "data": {
    "action": "rag_query|search_query|document_ingested|summarization",
    "framework": "haystack",
    "success": true,
    /* action-specific data */
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Framework Registry Architecture

### **AIFrameworkInterface**
Abstract interface that all frameworks must implement:
- `execute_rag_query()`
- `search_documents()`
- `ingest_document()`
- `summarize_text()`
- `list_pipelines()`
- `get_status()`

### **Framework Registration**
```python
from ai_framework_registry import ai_framework_registry

# Get framework
framework = ai_framework_registry.get_framework("haystack")

# List available frameworks
frameworks = ai_framework_registry.list_frameworks()

# Register custom framework
ai_framework_registry.register_framework(MyCustomFramework())
```

## Migration Guide

### **Old Endpoints → New Endpoints**
```
❌ POST /api/v1/haystack/rag     → ✅ POST /api/v1/rag
❌ POST /api/v1/haystack/search  → ✅ POST /api/v1/search
❌ POST /api/v1/haystack/documents → ✅ POST /api/v1/documents
❌ GET /api/v1/haystack/status   → ✅ GET /api/v1/ai/status
```

### **Request Changes**
Add `framework` field to all requests:
```json
// Old
{
  "query": "example"
}

// New
{
  "query": "example",
  "framework": "haystack"
}
```

## Framework Implementation Details

### **Haystack Framework**
- **Components**: Pipeline Manager, Document Store, Embedding Manager
- **Features**: RAG pipelines, BM25 retrieval, semantic search, document processing
- **Storage**: In-memory document store with optional vector databases
- **Dependencies**: `haystack-ai`, `sentence-transformers`, `transformers`

### **LlamaIndex Framework**
- **Components**: Index Manager, Document Processor, Query Engine
- **Features**: Vector indexes, query engines, document readers, summarization
- **Storage**: Persistent vector storage with configurable backends
- **Dependencies**: `llama-index`, `llama-index-core`, `llama-index-readers-file`

## Future Framework Integration

To add support for new frameworks (e.g., LangChain):

1. **Implement the Interface**
   ```python
   class LangChainFramework(AIFrameworkInterface):
       # Implement all required methods
   ```

2. **Register the Framework**
   ```python
   ai_framework_registry.register_framework(LangChainFramework())
   ```

3. **Update Documentation**
   - Add to supported frameworks list
   - Document framework-specific configuration options

## Benefits

✅ **Framework Flexibility**: Easy to add new AI frameworks
✅ **Backward Compatibility**: Default framework parameter maintains compatibility
✅ **Unified API**: Consistent interface across all frameworks
✅ **Future-Proof**: Architecture ready for LlamaIndex, LangChain, etc.
✅ **Configuration**: Framework-specific options via `framework_config`
✅ **Error Handling**: Clear error messages for unsupported operations
✅ **Monitoring**: Framework-specific metrics and status reporting

This design provides a solid foundation for supporting multiple AI frameworks while maintaining a clean, consistent API interface.
