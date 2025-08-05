"""
Search API Usage Examples

Demonstrates how to use the REST API search system with various
query patterns, filters, and advanced search capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests


class SearchAPIClient:
    """Client for interacting with the Search API."""

    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Authorization': f'Bearer {auth_token}', 'Content-Type': 'application/json'}

    def get_search_schema(self, entity_type: str) -> Dict[str, Any]:
        """Get search schema for entity type."""
        response = requests.get(f"{self.base_url}/api/search/schema/{entity_type}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def basic_search(
        self,
        entity_type: str,
        query: str = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = None,
        sort_order: str = 'desc',
    ) -> Dict[str, Any]:
        """Perform basic search."""
        params = {'page': page, 'page_size': page_size, 'sort_order': sort_order}

        if query:
            params['q'] = query
        if sort_by:
            params['sort_by'] = sort_by

        response = requests.get(f"{self.base_url}/api/search/{entity_type}", headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def advanced_search(self, entity_type: str, search_request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced search with filters."""
        response = requests.post(f"{self.base_url}/api/search/{entity_type}", headers=self.headers, json=search_request)
        response.raise_for_status()
        return response.json()

    def get_suggestions(self, entity_type: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """Get search suggestions."""
        params = {'q': query, 'limit': limit}
        response = requests.get(
            f"{self.base_url}/api/search/suggest/{entity_type}", headers=self.headers, params=params
        )
        response.raise_for_status()
        return response.json()

    def get_facets(self, entity_type: str, query: str = None) -> Dict[str, Any]:
        """Get search facets."""
        params = {}
        if query:
            params['q'] = query

        response = requests.get(f"{self.base_url}/api/search/facets/{entity_type}", headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()


def example_basic_searches():
    """Examples of basic search operations."""
    print("=== Basic Search Examples ===")

    # Initialize client (replace with actual URL and token)
    client = SearchAPIClient("http://localhost:8000", "your-auth-token")

    # Example 1: Simple text search
    print("\n1. Simple text search for users:")
    try:
        result = client.basic_search("users", query="admin")
        print(f"Found {result.get('total_count', 0)} users matching 'admin'")
        for user in result.get('items', [])[:3]:
            print(f"  - {user['username']} ({user['email']})")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Search with sorting
    print("\n2. Search documents sorted by size:")
    try:
        result = client.basic_search("documents", sort_by="size_bytes", sort_order="desc", page_size=5)
        print(f"Top 5 largest documents:")
        for doc in result.get('items', []):
            print(f"  - {doc['filename']}: {doc['size_bytes']} bytes")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Paginated search
    print("\n3. Paginated search through queries:")
    try:
        page = 1
        while page <= 3:  # Show first 3 pages
            result = client.basic_search("queries", page=page, page_size=5)
            print(f"Page {page}:")
            for query in result.get('items', []):
                print(f"  - {query['query_type']}: {query['query_text'][:50]}...")

            if not result.get('has_next', False):
                break
            page += 1
    except Exception as e:
        print(f"Error: {e}")


def example_advanced_searches():
    """Examples of advanced search with filters."""
    print("\n=== Advanced Search Examples ===")

    client = SearchAPIClient("http://localhost:8000", "your-auth-token")

    # Example 1: Search with equality filter
    print("\n1. Active API keys only:")
    search_request = {
        "filters": [{"field": "is_active", "operator": "eq", "value": True}],
        "sort_by": "usage_count",
        "sort_order": "desc",
    }

    try:
        result = client.advanced_search("api_keys", search_request)
        print(f"Found {len(result.get('items', []))} active API keys")
        for key in result.get('items', [])[:3]:
            print(f"  - {key['name']}: {key['usage_count']} uses")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Range filter for document size
    print("\n2. Large documents (>5MB):")
    search_request = {
        "filters": [
            {"field": "size_bytes", "operator": "gt", "value": 5242880},  # 5MB
            {"field": "processing_status", "operator": "eq", "value": "completed"},
        ]
    }

    try:
        result = client.advanced_search("documents", search_request)
        print(f"Found {len(result.get('items', []))} large completed documents")
        for doc in result.get('items', []):
            size_mb = doc['size_bytes'] / (1024 * 1024)
            print(f"  - {doc['filename']}: {size_mb:.1f}MB")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Text search with contains filter
    print("\n3. Queries containing 'search' with good performance:")
    search_request = {
        "query": "search",
        "filters": [
            {"field": "execution_time_ms", "operator": "lt", "value": 1000},  # Less than 1 second
            {"field": "success", "operator": "eq", "value": True},
        ],
        "sort_by": "execution_time_ms",
        "sort_order": "asc",
    }

    try:
        result = client.advanced_search("queries", search_request)
        print(f"Found {len(result.get('items', []))} fast successful search queries")
        for query in result.get('items', [])[:3]:
            print(f"  - {query['execution_time_ms']}ms: {query['query_text'][:60]}...")
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Date range filter
    print("\n4. Recent errors (last 24 hours) by severity:")
    yesterday = datetime.now() - timedelta(days=1)
    search_request = {
        "filters": [
            {"field": "created_at", "operator": "gte", "value": yesterday.isoformat()},
            {"field": "resolved", "operator": "eq", "value": False},
        ],
        "sort_by": "created_at",
        "sort_order": "desc",
    }

    try:
        result = client.advanced_search("errors", search_request)
        print(f"Found {len(result.get('items', []))} unresolved recent errors")
        for error in result.get('items', [])[:3]:
            print(f"  - {error['severity']}: {error['message'][:50]}...")
    except Exception as e:
        print(f"Error: {e}")

    # Example 5: Multiple value filter (IN operator)
    print("\n5. Queries of specific types:")
    search_request = {
        "filters": [{"field": "query_type", "operator": "in", "value": ["rag", "search", "summarization"]}]
    }

    try:
        result = client.advanced_search("queries", search_request)
        print(f"Found {len(result.get('items', []))} queries of specified types")

        # Count by type
        type_counts = {}
        for query in result.get('items', []):
            query_type = query['query_type']
            type_counts[query_type] = type_counts.get(query_type, 0) + 1

        for query_type, count in type_counts.items():
            print(f"  - {query_type}: {count}")
    except Exception as e:
        print(f"Error: {e}")


def example_search_features():
    """Examples of additional search features."""
    print("\n=== Search Features Examples ===")

    client = SearchAPIClient("http://localhost:8000", "your-auth-token")

    # Example 1: Get search schema
    print("\n1. Available search options for documents:")
    try:
        schema = client.get_search_schema("documents")
        print(f"Searchable fields: {', '.join(schema['searchable_fields'])}")
        print(f"Filterable fields: {', '.join(schema['filterable_fields'])}")
        print(f"Available operators: {', '.join(schema['available_operators'])}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Search suggestions
    print("\n2. Search suggestions for 'doc':")
    try:
        suggestions = client.get_suggestions("documents", "doc", limit=5)
        print("Suggestions:")
        for suggestion in suggestions['suggestions']:
            print(f"  - {suggestion}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Search facets
    print("\n3. Available facets for documents:")
    try:
        facets = client.get_facets("documents")
        for field, values in facets['facets'].items():
            print(f"{field}:")
            for item in values[:3]:  # Show top 3
                print(f"  - {item['value']}: {item['count']}")
    except Exception as e:
        print(f"Error: {e}")


def example_business_use_cases():
    """Examples of business-specific search use cases."""
    print("\n=== Business Use Cases ===")

    client = SearchAPIClient("http://localhost:8000", "your-auth-token")

    # Use Case 1: Find inactive users for cleanup
    print("\n1. Find inactive users for potential cleanup:")
    last_month = datetime.now() - timedelta(days=30)
    search_request = {
        "filters": [
            {"field": "last_login", "operator": "lt", "value": last_month.isoformat()},
            {"field": "is_active", "operator": "eq", "value": True},
        ],
        "sort_by": "last_login",
        "sort_order": "asc",
    }

    try:
        result = client.advanced_search("users", search_request)
        print(f"Found {len(result.get('items', []))} potentially inactive users")
        for user in result.get('items', [])[:5]:
            last_login = user.get('last_login', 'Never')
            print(f"  - {user['username']}: Last login {last_login}")
    except Exception as e:
        print(f"Error: {e}")

    # Use Case 2: Monitor system performance
    print("\n2. Find slow queries that need optimization:")
    search_request = {
        "filters": [
            {"field": "execution_time_ms", "operator": "gt", "value": 5000},  # >5 seconds
            {"field": "success", "operator": "eq", "value": True},
        ],
        "sort_by": "execution_time_ms",
        "sort_order": "desc",
        "page_size": 10,
    }

    try:
        result = client.advanced_search("queries", search_request)
        print(f"Found {len(result.get('items', []))} slow queries needing attention")
        for query in result.get('items', []):
            time_sec = query['execution_time_ms'] / 1000
            print(f"  - {time_sec:.1f}s: {query['query_text'][:60]}...")
    except Exception as e:
        print(f"Error: {e}")

    # Use Case 3: Audit API key usage
    print("\n3. Audit high-usage API keys:")
    search_request = {
        "filters": [
            {"field": "usage_count", "operator": "gt", "value": 1000},
            {"field": "is_active", "operator": "eq", "value": True},
        ],
        "sort_by": "usage_count",
        "sort_order": "desc",
    }

    try:
        result = client.advanced_search("api_keys", search_request)
        print(f"Found {len(result.get('items', []))} high-usage API keys")
        for key in result.get('items', []):
            print(f"  - {key['name']}: {key['usage_count']} uses")
    except Exception as e:
        print(f"Error: {e}")

    # Use Case 4: Error analysis
    print("\n4. Critical errors requiring immediate attention:")
    search_request = {
        "filters": [
            {"field": "severity", "operator": "eq", "value": "critical"},
            {"field": "resolved", "operator": "eq", "value": False},
        ],
        "sort_by": "created_at",
        "sort_order": "desc",
    }

    try:
        result = client.advanced_search("errors", search_request)
        print(f"Found {len(result.get('items', []))} unresolved critical errors")
        for error in result.get('items', []):
            print(f"  - {error['category']}: {error['message'][:60]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    """Run all examples."""
    print("MCP Platform Search API Examples")
    print("=" * 50)

    # Note: These examples require a running server and valid auth token
    print("Note: Update base_url and auth_token before running")

    example_basic_searches()
    example_advanced_searches()
    example_search_features()
    example_business_use_cases()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nFor more information, check the API documentation at:")
    print("http://localhost:8000/docs#/Search")
