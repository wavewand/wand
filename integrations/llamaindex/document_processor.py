"""
LlamaIndex Document Processing

This module handles document ingestion, parsing, and preprocessing for LlamaIndex.
"""

import hashlib
import io
import logging
import mimetypes
import uuid
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional

try:
    from llama_index.core import Document
    from llama_index.readers.file import DocxReader, PDFReader

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logging.warning("LlamaIndex not available")
    LLAMAINDEX_AVAILABLE = False

    class Document:
        def __init__(self, *args, **kwargs):
            pass

    class PDFReader:
        def load_data(self, *args, **kwargs):
            return []

    class DocxReader:
        def load_data(self, *args, **kwargs):
            return []


# Additional document processing
try:
    import markdown
    from bs4 import BeautifulSoup

    HTML_PROCESSING_AVAILABLE = True
    MARKDOWN_PROCESSING_AVAILABLE = True
except ImportError:
    HTML_PROCESSING_AVAILABLE = False
    MARKDOWN_PROCESSING_AVAILABLE = False


class LlamaIndexDocumentProcessor:
    """Processes documents for LlamaIndex integration."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_documents: Dict[str, Dict[str, Any]] = {}

        # Initialize readers
        if LLAMAINDEX_AVAILABLE:
            self.pdf_reader = PDFReader()
            self.docx_reader = DocxReader()
        else:
            self.pdf_reader = None
            self.docx_reader = None

    def process_document(
        self, filename: str, content: bytes, content_type: str = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process a document and return LlamaIndex Document objects."""
        if not LLAMAINDEX_AVAILABLE:
            return {"success": False, "error": "LlamaIndex not available"}

        try:
            # Determine content type
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
                content_type = content_type or "text/plain"

            # Generate document ID
            doc_id = str(uuid.uuid4())

            # Process based on content type
            documents = self._process_by_type(content, content_type, filename)

            if not documents:
                return {"success": False, "error": f"Failed to process document of type {content_type}"}

            # Add metadata to documents
            base_metadata = {
                "filename": filename,
                "content_type": content_type,
                "document_id": doc_id,
                "processed_at": datetime.now().isoformat(),
                "size_bytes": len(content),
                "content_hash": hashlib.sha256(content).hexdigest(),
                **(metadata or {}),
            }

            # Update document metadata
            for doc in documents:
                if hasattr(doc, 'metadata'):
                    doc.metadata.update(base_metadata)
                else:
                    doc.metadata = base_metadata

            # Store processed document info
            self.processed_documents[doc_id] = {
                "id": doc_id,
                "filename": filename,
                "content_type": content_type,
                "document_count": len(documents),
                "metadata": base_metadata,
                "created_at": datetime.now(),
            }

            self.logger.info(f"Processed document '{filename}' into {len(documents)} LlamaIndex documents")

            return {
                "success": True,
                "document_id": doc_id,
                "documents": documents,
                "document_count": len(documents),
                "metadata": base_metadata,
            }

        except Exception as e:
            self.logger.error(f"Failed to process document '{filename}': {e}")
            return {"success": False, "error": str(e)}

    def _process_by_type(self, content: bytes, content_type: str, filename: str) -> List[Document]:
        """Process content based on its type."""
        try:
            if content_type == "application/pdf":
                return self._process_pdf(content)
            elif content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                return self._process_docx(content)
            elif content_type in ["text/html", "application/xhtml+xml"]:
                return self._process_html(content)
            elif content_type == "text/markdown" or filename.endswith(('.md', '.markdown')):
                return self._process_markdown(content)
            elif content_type.startswith("text/") or content_type == "application/json":
                return self._process_text(content)
            else:
                # Try to process as text
                return self._process_text(content)

        except Exception as e:
            self.logger.error(f"Error processing {content_type}: {e}")
            return []

    def _process_pdf(self, content: bytes) -> List[Document]:
        """Process PDF content using LlamaIndex PDFReader."""
        if not self.pdf_reader:
            self.logger.warning("PDF reader not available")
            return []

        try:
            # Create a temporary file-like object
            pdf_file = io.BytesIO(content)

            # Use LlamaIndex PDFReader
            documents = self.pdf_reader.load_data(pdf_file)

            return documents

        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            return []

    def _process_docx(self, content: bytes) -> List[Document]:
        """Process DOCX content using LlamaIndex DocxReader."""
        if not self.docx_reader:
            self.logger.warning("DOCX reader not available")
            return []

        try:
            # Create a temporary file-like object
            docx_file = io.BytesIO(content)

            # Use LlamaIndex DocxReader
            documents = self.docx_reader.load_data(docx_file)

            return documents

        except Exception as e:
            self.logger.error(f"DOCX processing failed: {e}")
            return []

    def _process_html(self, content: bytes) -> List[Document]:
        """Process HTML content."""
        if not HTML_PROCESSING_AVAILABLE:
            self.logger.warning("HTML processing not available - install beautifulsoup4")
            return self._process_text(content)

        try:
            html_text = content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text_content = soup.get_text(separator='\n', strip=True)

            # Create LlamaIndex document
            document = Document(text=text_content)
            return [document]

        except Exception as e:
            self.logger.error(f"HTML processing failed: {e}")
            return []

    def _process_markdown(self, content: bytes) -> List[Document]:
        """Process Markdown content."""
        try:
            md_text = content.decode('utf-8', errors='ignore')

            if MARKDOWN_PROCESSING_AVAILABLE:
                # Convert to HTML first, then extract text
                html = markdown.markdown(md_text)

                if HTML_PROCESSING_AVAILABLE:
                    soup = BeautifulSoup(html, 'html.parser')
                    text_content = soup.get_text(separator='\n', strip=True)
                else:
                    text_content = md_text
            else:
                text_content = md_text

            # Create LlamaIndex document
            document = Document(text=text_content)
            return [document]

        except Exception as e:
            self.logger.error(f"Markdown processing failed: {e}")
            return []

    def _process_text(self, content: bytes) -> List[Document]:
        """Process plain text content."""
        try:
            text_content = content.decode('utf-8', errors='ignore')

            # Create LlamaIndex document
            document = Document(text=text_content)
            return [document]

        except Exception as e:
            self.logger.error(f"Text processing failed: {e}")
            return []

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document."""
        return self.processed_documents.get(document_id)

    def list_processed_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List processed documents."""
        documents = list(self.processed_documents.values())
        return documents[:limit]

    def delete_processed_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a processed document from tracking."""
        try:
            if document_id not in self.processed_documents:
                return {"success": False, "error": f"Document {document_id} not found"}

            del self.processed_documents[document_id]

            return {"success": True, "message": f"Document {document_id} removed from tracking"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get document processor statistics."""
        content_types = {}
        total_docs = 0

        for doc_info in self.processed_documents.values():
            content_type = doc_info.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
            total_docs += doc_info.get("document_count", 1)

        return {
            "total_processed_files": len(self.processed_documents),
            "total_documents": total_docs,
            "content_types": content_types,
            "processing_capabilities": {
                "pdf": bool(self.pdf_reader),
                "docx": bool(self.docx_reader),
                "html": HTML_PROCESSING_AVAILABLE,
                "markdown": MARKDOWN_PROCESSING_AVAILABLE,
                "text": True,
            },
            "llamaindex_available": LLAMAINDEX_AVAILABLE,
        }
