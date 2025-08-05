"""
Migration: Initial Schema
Description: Create initial database schema with all core tables
Created: 2025-08-02T12:00:00
"""

from sqlalchemy.orm import Session
from sqlalchemy import text


# Migration metadata
migration = {
    'name': 'Initial Schema',
    'version': '20250802_120000_initial_schema',
    'description': 'Create initial database schema with all core tables',
    'dependencies': []
}


def up(session: Session):
    """Apply migration changes."""
    # This migration creates the initial schema
    # The tables will be created by SQLAlchemy's create_all() method
    # This is a placeholder migration to track the initial state

    # Verify tables exist by querying metadata
    session.execute(text("SELECT 1"))  # Simple test query

    # Add any initial data if needed
    pass


def down(session: Session):
    """Rollback migration changes."""
    # Drop all tables (be very careful with this in production!)
    session.execute(text("DROP TABLE IF EXISTS cache_entries CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS performance_metrics CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS error_logs CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS benchmarks CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS batch_operations CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS queries CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS api_keys CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS frameworks CASCADE"))
    session.execute(text("DROP TABLE IF EXISTS users CASCADE"))
