"""
REST API Search and Filtering System

Provides comprehensive search, filtering, sorting, and pagination capabilities
across all platform entities with advanced query building and full-text search.
"""

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import and_, asc, desc, func, or_, text
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from database.connection import get_session
from database.models import APIKey, BatchOperation, Document, ErrorLog, Framework
from database.models import Query as QueryModel
from database.models import User
from database.repositories import (
    APIKeyRepository,
    BatchRepository,
    DocumentRepository,
    ErrorLogRepository,
    QueryRepository,
    UserRepository,
)
from observability.logging import get_logger
from security.auth import get_current_user


class SortOrder(str, Enum):
    """Sort order options."""

    ASC = "asc"
    DESC = "desc"


class SearchOperator(str, Enum):
    """Search operators for filtering."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"


@dataclass
class SearchFilter:
    """Individual search filter."""

    field: str
    operator: SearchOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class SearchParams:
    """Search parameters structure."""

    query: Optional[str] = None
    filters: List[SearchFilter] = None
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC
    page: int = 1
    page_size: int = 20
    include_count: bool = True

    def __post_init__(self):
        if self.filters is None:
            self.filters = []


@dataclass
class SearchResult:
    """Search result structure."""

    items: List[Dict[str, Any]]
    total_count: Optional[int] = None
    page: int = 1
    page_size: int = 20
    total_pages: Optional[int] = None
    has_next: bool = False
    has_previous: bool = False
    filters_applied: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.filters_applied is None:
            self.filters_applied = []

        if self.total_count is not None:
            self.total_pages = (self.total_count + self.page_size - 1) // self.page_size
            self.has_next = self.page < self.total_pages
            self.has_previous = self.page > 1


class SearchEngine:
    """Advanced search engine for all platform entities."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Repository mapping
        self.repositories = {
            'users': UserRepository(),
            'api_keys': APIKeyRepository(),
            'documents': DocumentRepository(),
            'queries': QueryRepository(),
            'batches': BatchRepository(),
            'errors': ErrorLogRepository(),
        }

        # Model mapping
        self.models = {
            'users': User,
            'api_keys': APIKey,
            'frameworks': Framework,
            'documents': Document,
            'queries': QueryModel,
            'batches': BatchOperation,
            'errors': ErrorLog,
        }

        # Searchable fields for each entity
        self.searchable_fields = {
            'users': ['username', 'email', 'role'],
            'api_keys': ['name', 'key_id'],
            'frameworks': ['name', 'display_name'],
            'documents': ['filename', 'content_type', 'document_id'],
            'queries': ['query_text', 'query_type'],
            'batches': ['batch_id', 'batch_type'],
            'errors': ['message', 'category', 'severity', 'operation'],
        }

        # Filterable fields for each entity
        self.filterable_fields = {
            'users': ['role', 'is_active', 'created_at', 'last_login'],
            'api_keys': ['is_active', 'expires_at', 'created_at', 'usage_count'],
            'frameworks': ['is_enabled', 'health_status', 'created_at'],
            'documents': ['content_type', 'processing_status', 'size_bytes', 'created_at'],
            'queries': ['query_type', 'success', 'execution_time_ms', 'created_at'],
            'batches': ['batch_type', 'status', 'total_items', 'created_at'],
            'errors': ['category', 'severity', 'resolved', 'created_at'],
        }

    def search(self, entity_type: str, params: SearchParams, session: Session) -> SearchResult:
        """Perform comprehensive search on specified entity type."""
        try:
            if entity_type not in self.models:
                raise ValueError(f"Unknown entity type: {entity_type}")

            model = self.models[entity_type]
            query = session.query(model)

            # Apply text search
            if params.query:
                query = self._apply_text_search(query, entity_type, params.query)

            # Apply filters
            for filter_obj in params.filters:
                query = self._apply_filter(query, model, filter_obj)

            # Get total count before pagination
            total_count = None
            if params.include_count:
                total_count = query.count()

            # Apply sorting
            if params.sort_by:
                query = self._apply_sorting(query, model, params.sort_by, params.sort_order)
            else:
                # Default sorting by created_at if available
                if hasattr(model, 'created_at'):
                    query = query.order_by(desc(model.created_at))

            # Apply pagination
            offset = (params.page - 1) * params.page_size
            query = query.offset(offset).limit(params.page_size)

            # Execute query
            items = query.all()

            # Convert to dictionaries
            result_items = []
            for item in items:
                if hasattr(item, 'to_dict'):
                    result_items.append(item.to_dict())
                else:
                    # Fallback to basic dict conversion
                    result_items.append({column.name: getattr(item, column.name) for column in item.__table__.columns})

            # Create search result
            result = SearchResult(
                items=result_items,
                total_count=total_count,
                page=params.page,
                page_size=params.page_size,
                filters_applied=[asdict(f) for f in params.filters],
            )

            self.logger.debug(f"Search completed: {entity_type}, {len(result_items)} items found")
            return result

        except Exception as e:
            self.logger.error(f"Search error for {entity_type}: {e}")
            raise

    def _apply_text_search(self, query: Select, entity_type: str, search_text: str) -> Select:
        """Apply full-text search across searchable fields."""
        searchable = self.searchable_fields.get(entity_type, [])
        if not searchable:
            return query

        model = self.models[entity_type]
        search_conditions = []

        # Create ILIKE conditions for each searchable field
        for field_name in searchable:
            if hasattr(model, field_name):
                field = getattr(model, field_name)
                search_conditions.append(field.ilike(f"%{search_text}%"))

        if search_conditions:
            query = query.filter(or_(*search_conditions))

        return query

    def _apply_filter(self, query: Select, model, filter_obj: SearchFilter) -> Select:
        """Apply individual filter to query."""
        if not hasattr(model, filter_obj.field):
            raise ValueError(f"Invalid filter field: {filter_obj.field}")

        field = getattr(model, filter_obj.field)

        # Apply operator
        if filter_obj.operator == SearchOperator.EQUALS:
            return query.filter(field == filter_obj.value)

        elif filter_obj.operator == SearchOperator.NOT_EQUALS:
            return query.filter(field != filter_obj.value)

        elif filter_obj.operator == SearchOperator.GREATER_THAN:
            return query.filter(field > filter_obj.value)

        elif filter_obj.operator == SearchOperator.GREATER_EQUAL:
            return query.filter(field >= filter_obj.value)

        elif filter_obj.operator == SearchOperator.LESS_THAN:
            return query.filter(field < filter_obj.value)

        elif filter_obj.operator == SearchOperator.LESS_EQUAL:
            return query.filter(field <= filter_obj.value)

        elif filter_obj.operator == SearchOperator.CONTAINS:
            if filter_obj.case_sensitive:
                return query.filter(field.like(f"%{filter_obj.value}%"))
            else:
                return query.filter(field.ilike(f"%{filter_obj.value}%"))

        elif filter_obj.operator == SearchOperator.STARTS_WITH:
            if filter_obj.case_sensitive:
                return query.filter(field.like(f"{filter_obj.value}%"))
            else:
                return query.filter(field.ilike(f"{filter_obj.value}%"))

        elif filter_obj.operator == SearchOperator.ENDS_WITH:
            if filter_obj.case_sensitive:
                return query.filter(field.like(f"%{filter_obj.value}"))
            else:
                return query.filter(field.ilike(f"%{filter_obj.value}"))

        elif filter_obj.operator == SearchOperator.IN:
            if isinstance(filter_obj.value, (list, tuple)):
                return query.filter(field.in_(filter_obj.value))
            else:
                return query.filter(field == filter_obj.value)

        elif filter_obj.operator == SearchOperator.NOT_IN:
            if isinstance(filter_obj.value, (list, tuple)):
                return query.filter(~field.in_(filter_obj.value))
            else:
                return query.filter(field != filter_obj.value)

        elif filter_obj.operator == SearchOperator.IS_NULL:
            return query.filter(field.is_(None))

        elif filter_obj.operator == SearchOperator.IS_NOT_NULL:
            return query.filter(field.is_not(None))

        elif filter_obj.operator == SearchOperator.BETWEEN:
            if isinstance(filter_obj.value, (list, tuple)) and len(filter_obj.value) == 2:
                return query.filter(field.between(filter_obj.value[0], filter_obj.value[1]))
            else:
                raise ValueError("BETWEEN operator requires a list/tuple of 2 values")

        else:
            raise ValueError(f"Unsupported operator: {filter_obj.operator}")

    def _apply_sorting(self, query: Select, model, sort_field: str, sort_order: SortOrder) -> Select:
        """Apply sorting to query."""
        if not hasattr(model, sort_field):
            raise ValueError(f"Invalid sort field: {sort_field}")

        field = getattr(model, sort_field)

        if sort_order == SortOrder.ASC:
            return query.order_by(asc(field))
        else:
            return query.order_by(desc(field))

    def get_search_schema(self, entity_type: str) -> Dict[str, Any]:
        """Get search schema for entity type."""
        if entity_type not in self.models:
            raise ValueError(f"Unknown entity type: {entity_type}")

        return {
            'entity_type': entity_type,
            'searchable_fields': self.searchable_fields.get(entity_type, []),
            'filterable_fields': self.filterable_fields.get(entity_type, []),
            'available_operators': [op.value for op in SearchOperator],
            'sort_orders': [order.value for order in SortOrder],
        }


# FastAPI Router
router = APIRouter(prefix="/api/search", tags=["Search"])
search_engine = SearchEngine()


@router.get("/schema/{entity_type}")
async def get_search_schema(entity_type: str, current_user: User = Depends(get_current_user)):
    """Get search schema for entity type."""
    try:
        schema = search_engine.get_search_schema(entity_type)
        return JSONResponse(content=schema)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{entity_type}")
async def search_entities(
    entity_type: str,
    q: Optional[str] = Query(None, description="Text search query"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    include_count: bool = Query(True, description="Include total count"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Search entities with text query and basic parameters."""
    try:
        params = SearchParams(
            query=q, sort_by=sort_by, sort_order=sort_order, page=page, page_size=page_size, include_count=include_count
        )

        result = search_engine.search(entity_type, params, session)
        return JSONResponse(content=asdict(result))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("/{entity_type}")
async def advanced_search(
    entity_type: str,
    search_request: Dict[str, Any],
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Advanced search with complex filters."""
    try:
        # Parse search parameters
        params = SearchParams(
            query=search_request.get('query'),
            sort_by=search_request.get('sort_by'),
            sort_order=SortOrder(search_request.get('sort_order', 'desc')),
            page=search_request.get('page', 1),
            page_size=min(search_request.get('page_size', 20), 100),
            include_count=search_request.get('include_count', True),
        )

        # Parse filters
        filters = search_request.get('filters', [])
        for filter_data in filters:
            filter_obj = SearchFilter(
                field=filter_data['field'],
                operator=SearchOperator(filter_data['operator']),
                value=filter_data['value'],
                case_sensitive=filter_data.get('case_sensitive', False),
            )
            params.filters.append(filter_obj)

        result = search_engine.search(entity_type, params, session)
        return JSONResponse(content=asdict(result))

    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Advanced search failed")


@router.get("/suggest/{entity_type}")
async def search_suggestions(
    entity_type: str,
    q: str = Query(..., min_length=2, description="Search query for suggestions"),
    limit: int = Query(10, ge=1, le=50, description="Number of suggestions"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get search suggestions for autocomplete."""
    try:
        if entity_type not in search_engine.models:
            raise ValueError(f"Unknown entity type: {entity_type}")

        model = search_engine.models[entity_type]
        searchable_fields = search_engine.searchable_fields.get(entity_type, [])

        suggestions = []

        # Get suggestions from each searchable field
        for field_name in searchable_fields:
            if hasattr(model, field_name):
                field = getattr(model, field_name)

                # Query for distinct values matching the search term
                query = session.query(field).filter(field.ilike(f"%{q}%")).distinct().limit(limit)

                values = [row[0] for row in query.all() if row[0]]
                suggestions.extend(values)

        # Remove duplicates and limit results
        unique_suggestions = list(set(suggestions))[:limit]

        return JSONResponse(content={'suggestions': unique_suggestions, 'query': q, 'entity_type': entity_type})

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Suggestions failed")


@router.get("/facets/{entity_type}")
async def get_search_facets(
    entity_type: str,
    q: Optional[str] = Query(None, description="Optional search query to filter facets"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Get search facets (aggregated filter values) for entity type."""
    try:
        if entity_type not in search_engine.models:
            raise ValueError(f"Unknown entity type: {entity_type}")

        model = search_engine.models[entity_type]
        filterable_fields = search_engine.filterable_fields.get(entity_type, [])

        facets = {}

        for field_name in filterable_fields:
            if hasattr(model, field_name):
                field = getattr(model, field_name)

                # Build base query
                query = session.query(field, func.count(field)).group_by(field)

                # Apply text search if provided
                if q:
                    searchable_fields = search_engine.searchable_fields.get(entity_type, [])
                    search_conditions = []

                    for search_field in searchable_fields:
                        if hasattr(model, search_field):
                            search_field_obj = getattr(model, search_field)
                            search_conditions.append(search_field_obj.ilike(f"%{q}%"))

                    if search_conditions:
                        query = query.filter(or_(*search_conditions))

                # Execute query and format results
                results = query.all()
                facets[field_name] = [{'value': value, 'count': count} for value, count in results if value is not None]

        return JSONResponse(content={'facets': facets, 'entity_type': entity_type, 'query': q})

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Facets retrieval failed")
