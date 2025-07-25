"""Database models and persistence layer for Python API server."""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, JSON, ARRAY, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default='idle')
    capabilities = Column(ARRAY(String))
    current_task = Column(String(255))
    metrics = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_active = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationship
    tasks = relationship("Task", back_populates="agent")


class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(String(255), primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(String)
    type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default='pending')
    priority = Column(String(20), nullable=False, default='medium')
    assigned_to = Column(String(255), ForeignKey('agents.id', ondelete='SET NULL'))
    project_id = Column(String(255), ForeignKey('projects.id', ondelete='CASCADE'))
    dependencies = Column(ARRAY(String))
    result = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    project = relationship("Project", back_populates="tasks")


class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String)
    status = Column(String(50), nullable=False, default='planning')
    components = Column(ARRAY(String))
    progress = Column(Integer, default=0)
    tasks_total = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    agents = Column(ARRAY(String))
    start_date = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    due_date = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationship
    tasks = relationship("Task", back_populates="project")


class Integration(Base):
    __tablename__ = 'integrations'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    enabled = Column(Boolean, default=False)
    status = Column(String(50), nullable=False, default='disconnected')
    config = Column(JSON, default={})
    last_sync = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))


class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(255), nullable=False)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection string. If None, uses in-memory SQLite.
        """
        if database_url and database_url != "memory://":
            self.engine = create_engine(database_url, pool_size=10, max_overflow=20)
            self.is_postgres = True
        else:
            # Use in-memory SQLite for testing/development
            self.engine = create_engine("sqlite:///:memory:")
            self.is_postgres = False
        
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Ensure default integrations exist
        self.ensure_integrations()
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def ensure_integrations(self):
        """Ensure default integrations exist in the database."""
        default_integrations = [
            {"id": "slack", "name": "Slack", "type": "messaging"},
            {"id": "git", "name": "Git", "type": "version_control"},
            {"id": "jenkins", "name": "Jenkins", "type": "ci_cd"},
            {"id": "youtrack", "name": "YouTrack", "type": "issue_tracking"},
            {"id": "aws", "name": "AWS", "type": "cloud"},
            {"id": "postgresql", "name": "PostgreSQL", "type": "database"},
            {"id": "bambu", "name": "Bambu 3D", "type": "hardware"},
            {"id": "web_search", "name": "Web Search", "type": "search"},
        ]
        
        with self.session_scope() as session:
            for integration_data in default_integrations:
                existing = session.query(Integration).filter_by(id=integration_data["id"]).first()
                if not existing:
                    integration = Integration(**integration_data)
                    session.add(integration)
    
    # Agent operations
    def create_agent(self, agent_data: Dict[str, Any]) -> Agent:
        with self.session_scope() as session:
            agent = Agent(**agent_data)
            session.add(agent)
            session.flush()
            session.refresh(agent)
            return agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self.session_scope() as session:
            return session.query(Agent).filter_by(id=agent_id).first()
    
    def get_agents(self, status: Optional[str] = None, agent_type: Optional[str] = None) -> List[Agent]:
        with self.session_scope() as session:
            query = session.query(Agent)
            if status:
                query = query.filter_by(status=status)
            if agent_type:
                query = query.filter_by(type=agent_type)
            return query.order_by(Agent.created_at.desc()).all()
    
    def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Agent]:
        with self.session_scope() as session:
            agent = session.query(Agent).filter_by(id=agent_id).first()
            if agent:
                for key, value in update_data.items():
                    setattr(agent, key, value)
                agent.updated_at = datetime.now(timezone.utc)
                session.flush()
                session.refresh(agent)
            return agent
    
    def delete_agent(self, agent_id: str) -> bool:
        with self.session_scope() as session:
            agent = session.query(Agent).filter_by(id=agent_id).first()
            if agent:
                session.delete(agent)
                return True
            return False
    
    # Task operations
    def create_task(self, task_data: Dict[str, Any]) -> Task:
        with self.session_scope() as session:
            task = Task(**task_data)
            session.add(task)
            
            # Update project task count
            if task.project_id:
                project = session.query(Project).filter_by(id=task.project_id).first()
                if project:
                    project.tasks_total += 1
            
            session.flush()
            session.refresh(task)
            return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        with self.session_scope() as session:
            return session.query(Task).filter_by(id=task_id).first()
    
    def get_tasks(self, status: Optional[str] = None, 
                  priority: Optional[str] = None,
                  assigned_to: Optional[str] = None) -> List[Task]:
        with self.session_scope() as session:
            query = session.query(Task)
            if status:
                query = query.filter_by(status=status)
            if priority:
                query = query.filter_by(priority=priority)
            if assigned_to:
                query = query.filter_by(assigned_to=assigned_to)
            return query.order_by(Task.created_at.desc()).all()
    
    def update_task(self, task_id: str, update_data: Dict[str, Any]) -> Optional[Task]:
        with self.session_scope() as session:
            task = session.query(Task).filter_by(id=task_id).first()
            if task:
                old_status = task.status
                
                for key, value in update_data.items():
                    setattr(task, key, value)
                
                task.updated_at = datetime.now(timezone.utc)
                
                # Handle status changes
                new_status = update_data.get('status', old_status)
                if new_status == 'completed' and old_status != 'completed':
                    task.completed_at = datetime.now(timezone.utc)
                    
                    # Update project progress
                    if task.project_id:
                        project = session.query(Project).filter_by(id=task.project_id).first()
                        if project:
                            project.tasks_completed += 1
                            if project.tasks_total > 0:
                                project.progress = int((project.tasks_completed / project.tasks_total) * 100)
                elif old_status == 'completed' and new_status != 'completed':
                    task.completed_at = None
                    
                    # Update project progress
                    if task.project_id:
                        project = session.query(Project).filter_by(id=task.project_id).first()
                        if project:
                            project.tasks_completed = max(0, project.tasks_completed - 1)
                            if project.tasks_total > 0:
                                project.progress = int((project.tasks_completed / project.tasks_total) * 100)
                
                session.flush()
                session.refresh(task)
            return task
    
    def delete_task(self, task_id: str) -> bool:
        with self.session_scope() as session:
            task = session.query(Task).filter_by(id=task_id).first()
            if task:
                # Update project counts
                if task.project_id:
                    project = session.query(Project).filter_by(id=task.project_id).first()
                    if project:
                        project.tasks_total = max(0, project.tasks_total - 1)
                        if task.status == 'completed':
                            project.tasks_completed = max(0, project.tasks_completed - 1)
                        if project.tasks_total > 0:
                            project.progress = int((project.tasks_completed / project.tasks_total) * 100)
                        else:
                            project.progress = 0
                
                session.delete(task)
                return True
            return False
    
    # Project operations
    def create_project(self, project_data: Dict[str, Any]) -> Project:
        with self.session_scope() as session:
            project = Project(**project_data)
            session.add(project)
            session.flush()
            session.refresh(project)
            return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        with self.session_scope() as session:
            return session.query(Project).filter_by(id=project_id).first()
    
    def get_projects(self, status: Optional[str] = None) -> List[Project]:
        with self.session_scope() as session:
            query = session.query(Project)
            if status:
                query = query.filter_by(status=status)
            return query.order_by(Project.created_at.desc()).all()
    
    # Integration operations
    def get_integrations(self) -> List[Integration]:
        with self.session_scope() as session:
            return session.query(Integration).all()
    
    def update_integration(self, integration_id: str, enabled: bool, status: str) -> Optional[Integration]:
        with self.session_scope() as session:
            integration = session.query(Integration).filter_by(id=integration_id).first()
            if integration:
                integration.enabled = enabled
                integration.status = status
                integration.last_sync = datetime.now(timezone.utc)
                integration.updated_at = datetime.now(timezone.utc)
                session.flush()
                session.refresh(integration)
            return integration
    
    # Event logging
    def log_event(self, event_type: str, entity_type: str, entity_id: str, data: Dict[str, Any]):
        with self.session_scope() as session:
            event = Event(
                type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                data=data
            )
            session.add(event)