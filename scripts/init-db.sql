-- MCP Distributed System Database Schema
-- This script initializes the PostgreSQL database for the MCP system

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types
CREATE TYPE agent_type AS ENUM ('manager', 'frontend', 'backend', 'database', 'devops', 'integration', 'qa');
CREATE TYPE agent_status AS ENUM ('online', 'offline', 'busy', 'idle');
CREATE TYPE task_status AS ENUM ('pending', 'assigned', 'in_progress', 'completed', 'failed', 'blocked');
CREATE TYPE task_priority AS ENUM ('critical', 'high', 'medium', 'low');

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type agent_type NOT NULL,
    capabilities JSONB DEFAULT '{}',
    current_tasks TEXT[] DEFAULT ARRAY[]::TEXT[],
    status agent_status DEFAULT 'offline',
    max_concurrent_tasks INTEGER DEFAULT 5,
    performance_metrics JSONB DEFAULT '{}',
    port INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT,
    tech_stack JSONB DEFAULT '{}',
    priority task_priority DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) NOT NULL,
    priority task_priority DEFAULT 'medium',
    status task_status DEFAULT 'pending',
    assigned_to VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    dependencies TEXT[] DEFAULT ARRAY[]::TEXT[],
    subtasks TEXT[] DEFAULT ARRAY[]::TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Task history table for audit trail
CREATE TABLE IF NOT EXISTS task_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    agent_id VARCHAR(255) REFERENCES agents(id) ON DELETE SET NULL,
    previous_status task_status,
    new_status task_status,
    change_reason TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Integration logs table
CREATE TABLE IF NOT EXISTS integration_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    integration_name VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    parameters JSONB DEFAULT '{}',
    success BOOLEAN NOT NULL,
    result_data JSONB DEFAULT '{}',
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metric_unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_to ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_task_history_task_id ON task_history(task_id);
CREATE INDEX IF NOT EXISTS idx_integration_logs_name ON integration_logs(integration_name);
CREATE INDEX IF NOT EXISTS idx_integration_logs_created_at ON integration_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data for testing (optional)
INSERT INTO agents (id, name, type, capabilities, port) VALUES
    ('manager_50100', 'Manager Agent', 'manager', '{"planning": true, "coordination": true, "reporting": true, "risk_assessment": true}', 50100),
    ('frontend_50101', 'Frontend Agent', 'frontend', '{"react": true, "vue": true, "angular": true, "typescript": true, "css": true, "responsive_design": true}', 50101),
    ('backend_50102', 'Backend Agent', 'backend', '{"python": true, "go": true, "nodejs": true, "api_design": true, "microservices": true}', 50102),
    ('database_50103', 'Database Agent', 'database', '{"postgresql": true, "mysql": true, "mongodb": true, "redis": true, "optimization": true, "database_design": true}', 50103),
    ('devops_50104', 'DevOps Agent', 'devops', '{"aws": true, "docker": true, "kubernetes": true, "terraform": true, "jenkins": true, "monitoring": true}', 50104),
    ('integration_50105', 'Integration Agent', 'integration', '{"slack": true, "git": true, "webhooks": true, "api_integration": true, "youtrack": true}', 50105),
    ('qa_50106', 'QA Agent', 'qa', '{"testing": true, "automation": true, "selenium": true, "pytest": true, "quality_assurance": true}', 50106)
ON CONFLICT (id) DO NOTHING;

-- Grant permissions to mcp_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mcp_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mcp_user;
