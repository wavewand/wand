# Multi-Framework AI Implementation

## 🎯 Overview

The MCP system now supports multiple AI frameworks through a unified, framework-agnostic API. Users can seamlessly switch between different AI frameworks using the same endpoints by simply specifying the desired framework in their requests.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REST API Gateway                             │
│                 (Framework Agnostic)                           │
├─────────────────────────────────────────────────────────────────┤
│                AI Framework Registry                            │
│               (Plugin Architecture)                            │
├─────────────────────────────────────────────────────────────────┤
│  Haystack Framework  │  LlamaIndex Framework  │  Future...     │
│  ├─ Pipeline Mgr     │  ├─ Index Manager      │                │
│  ├─ Document Store   │  ├─ Document Processor │                │
│  └─ Embedding Mgr    │  └─ Query Engine       │                │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies for both frameworks
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export HUGGING_FACE_API_TOKEN="your-hf-token"  # Optional
```

### Basic Usage

```python
import aiohttp
import asyncio

async def example_usage():
    headers = {"x-api-key": "your-api-key", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:

        # 1. Ingest document with Haystack
        payload = {
            "filename": "example.txt",
            "content": "Your document content here...",
            "framework": "haystack",
            "content_type": "text/plain"
        }

        async with session.post("http://localhost:8000/api/v1/documents",
                               headers=headers, json=payload) as response:
            result = await response.json()
            print(f"Document ingested: {result}")

        # 2. Same document with LlamaIndex
        payload["framework"] = "llamaindex"

        async with session.post("http://localhost:8000/api/v1/documents",
                               headers=headers, json=payload) as response:
            result = await response.json()
            print(f"Document ingested: {result}")

        # 3. Compare RAG responses
        rag_payload = {
            "query": "What is this document about?",
            "framework": "haystack",  # Try both "haystack" and "llamaindex"
            "temperature": 0.7
        }

        async with session.post("http://localhost:8000/api/v1/rag",
                               headers=headers, json=rag_payload) as response:
            result = await response.json()
            print(f"RAG Response: {result}")

# Run the example
asyncio.run(example_usage())
```

## 🔧 API Endpoints

All endpoints support framework selection via the `framework` parameter:

| Endpoint | Method | Description | Frameworks |
|----------|--------|-------------|------------|
| `/api/v1/rag` | POST | RAG queries | ✅ Haystack, ✅ LlamaIndex |
| `/api/v1/search` | POST | Document search | ✅ Haystack, ✅ LlamaIndex |
| `/api/v1/documents` | POST/GET/DELETE | Document management | ✅ Haystack, ✅ LlamaIndex |
| `/api/v1/summarize` | POST | Text summarization | ✅ Haystack, ✅ LlamaIndex |
| `/api/v1/pipelines` | GET | List pipelines/indexes | ✅ Haystack, ✅ LlamaIndex |
| `/api/v1/ai/frameworks` | GET | List available frameworks | N/A |
| `/api/v1/ai/status` | GET | Framework health status | N/A |

## 🤖 Framework Comparison

### Haystack vs LlamaIndex

| Feature | Haystack | LlamaIndex |
|---------|----------|------------|
| **Focus** | Production pipelines | Research & experimentation |
| **Architecture** | Pipeline-based | Index-centric |
| **Storage** | In-memory + Vector DBs | Persistent vector storage |
| **Customization** | High (component-based) | Very high (query engines) |
| **Performance** | Optimized for scale | Optimized for flexibility |
| **Learning Curve** | Moderate | Gentle |

### When to Use Which?

**Choose Haystack when:**
- Building production RAG systems
- Need robust pipeline orchestration
- Require enterprise-grade features
- Working with large document collections

**Choose LlamaIndex when:**
- Prototyping AI applications quickly
- Need flexible query interfaces
- Working with diverse data sources
- Prioritizing ease of use

## 📊 Testing

### Run Comprehensive Tests

```bash
# Start the API server
python main.py

# Run multi-framework tests
python test_multi_framework_api.py

# Run framework comparison
python examples/framework_comparison_example.py
```

### Test Results Example

```
🎯 MULTI-FRAMEWORK API TEST SUMMARY
================================================================================
📊 Total Tests: 12
✅ Successful: 10
❌ Failed: 2
📈 Success Rate: 83.3%

📋 RAG Queries:
   ✅ haystack: Success
   ✅ llamaindex: Success

📋 Document Search:
   ✅ haystack: Success
   ✅ llamaindex: Success
```

## 🔍 Monitoring & Debugging

### Check Framework Status

```bash
curl -H "x-api-key: your-key" http://localhost:8000/api/v1/ai/status
```

### WebSocket Events

Connect to `ws://localhost:8000/ws` to receive real-time events:

```json
{
  "type": "ai_event",
  "data": {
    "action": "rag_query",
    "framework": "llamaindex",
    "success": true,
    "execution_time_ms": 234
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Framework Not Available**
   ```json
   {"success": false, "error": "Framework 'llamaindex' not supported"}
   ```
   **Solution**: Check dependencies and framework registration

2. **OpenAI API Key Missing**
   ```json
   {"success": false, "error": "OpenAI API key not configured"}
   ```
   **Solution**: Set `OPENAI_API_KEY` environment variable

3. **Document Ingestion Fails**
   ```json
   {"success": false, "error": "Failed to process document of type application/pdf"}
   ```
   **Solution**: Install document processing dependencies (`pypdf`, `python-docx`)

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for LLM operations
OPENAI_API_KEY=sk-...

# Optional configurations
HUGGING_FACE_API_TOKEN=hf_...
HAYSTACK_TELEMETRY_ENABLED=false

# Framework-specific storage
LLAMAINDEX_STORAGE_DIR=./storage/llamaindex
HAYSTACK_CACHE_DIR=./storage/haystack
```

### Framework-Specific Config

Pass custom configuration via `framework_config`:

```json
{
  "query": "Example query",
  "framework": "llamaindex",
  "framework_config": {
    "similarity_top_k": 10,
    "response_mode": "tree_summarize",
    "temperature": 0.9
  }
}
```

## 🚧 Extending with New Frameworks

### Add LangChain Support

1. **Create Integration Module**
   ```python
   # langchain_integration/
   # ├── __init__.py
   # ├── chain_manager.py
   # ├── vector_store.py
   # └── embeddings.py
   ```

2. **Implement Framework Interface**
   ```python
   class LangChainFramework(AIFrameworkInterface):
       async def execute_rag_query(self, query: str, **kwargs):
           # Implementation here
           pass
   ```

3. **Register Framework**
   ```python
   ai_framework_registry.register_framework(LangChainFramework())
   ```

## 📈 Performance Tips

1. **Pre-warm Indexes**: Ingest documents before running queries
2. **Batch Operations**: Use bulk document ingestion when possible
3. **Cache Embeddings**: Enable embedding caching for repeated queries
4. **Monitor Resources**: Watch memory usage with large document collections
5. **Framework Selection**: Choose the right framework for your use case

## 🎯 Next Steps

- [ ] Add LangChain framework support
- [ ] Implement custom vector database backends
- [ ] Add batch processing endpoints
- [ ] Create framework benchmarking tools
- [ ] Add advanced query analytics

## 📚 Resources

- [API Design Documentation](./API_DESIGN.md)
- [Haystack Documentation](https://haystack.deepset.ai/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Example Scripts](./examples/)
- [Test Suite](./test_multi_framework_api.py)

---

🎉 **The MCP system now provides a unified, framework-agnostic API for AI operations, making it easy to compare, switch, and extend different AI frameworks!**
