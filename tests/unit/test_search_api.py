"""
Unit Tests for REST API Search System

Tests for search endpoints, filtering, pagination, and advanced search functionality.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.search import SearchEngine, SearchFilter, SearchOperator, SearchParams, SortOrder
from config.settings import DatabaseSettings
from database.init import DatabaseInitializer
from database.models import APIKey, Document, Framework, Query, User
from security.auth import create_access_token, hash_password


@pytest.fixture
def db_initializer():
    """Create test database."""
    settings = DatabaseSettings(url="sqlite:///:memory:", echo=False)
    initializer = DatabaseInitializer(settings)
    initializer.initialize(run_migrations=False, create_tables=True)
    yield initializer
    initializer.cleanup()


@pytest.fixture
def db_session(db_initializer):
    """Create database session."""
    with db_initializer.db_manager.get_session() as session:
        yield session


@pytest.fixture
def test_user(db_session):
    """Create test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=hash_password("password123"),
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_token(test_user):
    """Create authentication token."""
    return create_access_token({"sub": test_user.id})


@pytest.fixture
def sample_data(db_session, test_user):
    """Create sample data for testing."""
    # Create a framework first
    framework = Framework(
        id="framework_1",
        name="test_framework",
        display_name="Test Framework",
        is_enabled=True,
        configuration={"type": "search"},
    )
    db_session.add(framework)
    db_session.flush()  # Ensure framework is created before documents

    # Create API keys
    api_keys = []
    for i in range(5):
        api_key = APIKey(
            key_id=f"test_key_{i}",
            key_hash=f"hash_{i}",
            name=f"Test Key {i}",
            user_id=test_user.id,
            permissions=["read", "write"] if i < 3 else ["read"],
            is_active=i < 3,  # First 3 are active
            usage_count=i * 10,
        )
        api_keys.append(api_key)
        db_session.add(api_key)

    # Create documents
    documents = []
    for i in range(10):
        doc = Document(
            document_id=f"doc_{i}",
            filename=f"document_{i}.pdf",
            content_type="application/pdf",
            content_hash=f"hash_{i}",
            size_bytes=1000 + i * 100,
            framework_id="framework_1",
            user_id=test_user.id,
            processing_status="completed" if i < 7 else "pending",
        )
        documents.append(doc)
        db_session.add(doc)

    # Create queries
    queries = []
    for i in range(15):
        query = Query(
            query_text=f"Test query {i} about search functionality",
            query_hash=f"query_hash_{i}",
            query_type="search" if i % 2 == 0 else "rag",
            framework_id="framework_1",
            user_id=test_user.id,
            execution_time_ms=100 + i * 50,
            success=i < 12,  # Most are successful
            created_at=datetime.now(timezone.utc) - timedelta(hours=i),
        )
        queries.append(query)
        db_session.add(query)

    db_session.commit()
    return {'api_keys': api_keys, 'documents': documents, 'queries': queries}


class TestSearchEngine:
    """Test SearchEngine functionality."""

    def test_search_engine_initialization(self):
        """Test search engine initialization."""
        engine = SearchEngine()

        assert 'users' in engine.models
        assert 'api_keys' in engine.models
        assert 'documents' in engine.models
        assert 'queries' in engine.models

        assert 'users' in engine.searchable_fields
        assert 'username' in engine.searchable_fields['users']
        assert 'email' in engine.searchable_fields['users']

    def test_get_search_schema(self):
        """Test search schema retrieval."""
        engine = SearchEngine()
        schema = engine.get_search_schema('users')

        assert schema['entity_type'] == 'users'
        assert 'username' in schema['searchable_fields']
        assert 'email' in schema['searchable_fields']
        assert 'role' in schema['filterable_fields']
        assert 'eq' in schema['available_operators']
        assert 'asc' in schema['sort_orders']

    def test_basic_search(self, db_session, sample_data):
        """Test basic text search."""
        engine = SearchEngine()

        params = SearchParams(query="Test Key", page=1, page_size=10)

        result = engine.search('api_keys', params, db_session)

        assert len(result.items) <= 10
        assert result.page == 1
        assert result.page_size == 10
        assert result.total_count is not None

        # Check that results contain the search term
        for item in result.items:
            assert 'Test Key' in item['name']

    def test_search_with_filters(self, db_session, sample_data):
        """Test search with filters."""
        engine = SearchEngine()

        # Filter for active API keys only
        filter_obj = SearchFilter(field='is_active', operator=SearchOperator.EQUALS, value=True)

        params = SearchParams(filters=[filter_obj], page=1, page_size=10)

        result = engine.search('api_keys', params, db_session)

        # Should only return active keys (first 3)
        assert len(result.items) == 3
        for item in result.items:
            assert item['is_active'] is True

    def test_search_with_sorting(self, db_session, sample_data):
        """Test search with sorting."""
        engine = SearchEngine()

        params = SearchParams(sort_by='usage_count', sort_order=SortOrder.DESC, page=1, page_size=10)

        result = engine.search('api_keys', params, db_session)

        # Check that results are sorted by usage_count descending
        usage_counts = [item['usage_count'] for item in result.items]
        assert usage_counts == sorted(usage_counts, reverse=True)

    def test_search_pagination(self, db_session, sample_data):
        """Test search pagination."""
        engine = SearchEngine()

        # First page
        params = SearchParams(page=1, page_size=3)
        result = engine.search('documents', params, db_session)

        assert len(result.items) == 3
        assert result.page == 1
        assert result.has_next is True
        assert result.has_previous is False

        # Second page
        params = SearchParams(page=2, page_size=3)
        result = engine.search('documents', params, db_session)

        assert len(result.items) == 3
        assert result.page == 2
        assert result.has_previous is True

    def test_complex_filters(self, db_session, sample_data):
        """Test complex filtering scenarios."""
        engine = SearchEngine()

        # Multiple filters: completed documents with size > 1200
        filters = [
            SearchFilter(field='processing_status', operator=SearchOperator.EQUALS, value='completed'),
            SearchFilter(field='size_bytes', operator=SearchOperator.GREATER_THAN, value=1200),
        ]

        params = SearchParams(filters=filters)
        result = engine.search('documents', params, db_session)

        for item in result.items:
            assert item['processing_status'] == 'completed'
            assert item['size_bytes'] > 1200

    def test_search_operators(self, db_session, sample_data):
        """Test different search operators."""
        engine = SearchEngine()

        # Test CONTAINS operator
        filter_obj = SearchFilter(field='filename', operator=SearchOperator.CONTAINS, value='document')
        params = SearchParams(filters=[filter_obj])
        result = engine.search('documents', params, db_session)

        assert len(result.items) > 0
        for item in result.items:
            assert 'document' in item['filename'].lower()

        # Test IN operator
        filter_obj = SearchFilter(field='query_type', operator=SearchOperator.IN, value=['search', 'rag'])
        params = SearchParams(filters=[filter_obj])
        result = engine.search('queries', params, db_session)

        for item in result.items:
            assert item['query_type'] in ['search', 'rag']

    def test_search_with_invalid_entity(self, db_session):
        """Test search with invalid entity type."""
        engine = SearchEngine()

        params = SearchParams(query="test")

        with pytest.raises(ValueError, match="Unknown entity type"):
            engine.search('invalid_entity', params, db_session)

    def test_search_with_invalid_filter_field(self, db_session, sample_data):
        """Test search with invalid filter field."""
        engine = SearchEngine()

        filter_obj = SearchFilter(field='nonexistent_field', operator=SearchOperator.EQUALS, value='test')
        params = SearchParams(filters=[filter_obj])

        with pytest.raises(ValueError, match="Invalid filter field"):
            engine.search('users', params, db_session)


@pytest.mark.skipif(True, reason="API endpoint tests require FastAPI app - not available yet")
class TestSearchAPI:
    """Test search API endpoints."""

    @pytest.fixture
    def client(self, db_initializer):
        """Create test client."""
        from main import app

        # Override database dependency

        def override_get_session():
            with db_initializer.db_manager.get_session() as session:
                yield session

        app.dependency_overrides[get_session] = override_get_session
        return TestClient(app)

    def test_get_search_schema_endpoint(self, client, auth_token):
        """Test search schema endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/schema/users", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert data['entity_type'] == 'users'
        assert 'searchable_fields' in data
        assert 'filterable_fields' in data
        assert 'available_operators' in data

    def test_basic_search_endpoint(self, client, auth_token, sample_data):
        """Test basic search endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/api_keys?q=Test&page=1&page_size=5", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert 'items' in data
        assert 'total_count' in data
        assert 'page' in data
        assert data['page'] == 1
        assert len(data['items']) <= 5

    def test_advanced_search_endpoint(self, client, auth_token, sample_data):
        """Test advanced search endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}

        search_request = {
            "query": "document",
            "filters": [{"field": "processing_status", "operator": "eq", "value": "completed"}],
            "sort_by": "size_bytes",
            "sort_order": "desc",
            "page": 1,
            "page_size": 5,
        }

        response = client.post("/api/search/documents", json=search_request, headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert 'items' in data
        assert len(data['filters_applied']) == 1

        # Check filtering worked
        for item in data['items']:
            assert item['processing_status'] == 'completed'

    def test_search_suggestions_endpoint(self, client, auth_token, sample_data):
        """Test search suggestions endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/suggest/documents?q=doc&limit=5", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert 'suggestions' in data
        assert data['query'] == 'doc'
        assert data['entity_type'] == 'documents'
        assert len(data['suggestions']) <= 5

    def test_search_facets_endpoint(self, client, auth_token, sample_data):
        """Test search facets endpoint."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/facets/documents", headers=headers)

        assert response.status_code == 200
        data = response.json()

        assert 'facets' in data
        assert data['entity_type'] == 'documents'

        # Check that facets contain expected fields
        if 'processing_status' in data['facets']:
            facet_values = [item['value'] for item in data['facets']['processing_status']]
            assert 'completed' in facet_values or 'pending' in facet_values

    def test_unauthorized_search(self, client):
        """Test search without authentication."""
        response = client.get("/api/search/users")
        assert response.status_code == 401

    def test_search_invalid_entity_type(self, client, auth_token):
        """Test search with invalid entity type."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/invalid_entity", headers=headers)

        assert response.status_code == 400

    def test_search_with_large_page_size(self, client, auth_token):
        """Test search with page size exceeding limit."""
        headers = {"Authorization": f"Bearer {auth_token}"}
        response = client.get("/api/search/users?page_size=200", headers=headers)  # Exceeds limit of 100

        assert response.status_code == 200
        data = response.json()
        # Should be capped at 100
        assert data['page_size'] <= 100


class TestSearchPerformance:
    """Test search performance and optimization."""

    def test_search_with_large_dataset(self, db_session, test_user):
        """Test search performance with large dataset."""
        pytest.skip("Large dataset test requires significant resources - skipping for CI")

    def test_complex_query_performance(self, db_session, sample_data):
        """Test performance of complex queries with multiple filters."""
        engine = SearchEngine()

        filters = [
            SearchFilter('processing_status', SearchOperator.EQUALS, 'completed'),
            SearchFilter('size_bytes', SearchOperator.GREATER_THAN, 1000),
            SearchFilter('content_type', SearchOperator.CONTAINS, 'pdf'),
        ]

        params = SearchParams(query="document", filters=filters, sort_by="size_bytes", sort_order=SortOrder.DESC)

        import time

        start_time = time.time()

        result = engine.search('documents', params, db_session)

        end_time = time.time()
        search_time = end_time - start_time

        # Complex query should still be reasonably fast
        assert search_time < 0.5  # 500ms
        assert isinstance(result.items, list)
