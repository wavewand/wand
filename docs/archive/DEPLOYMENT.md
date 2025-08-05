# MCP Distributed System - Deployment Guide

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Make (optional but recommended)
- 8GB+ RAM for full system
- Linux/macOS/Windows

### 2. One-Command Setup
```bash
# Complete setup and start
make build && make start
```

### 3. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Generate gRPC code
python scripts/generate_grpc.py

# Start system
python main.py
```

## 📋 Available Commands

### System Management
```bash
make start          # Start the distributed system
make debug          # Start with debug logging
make status         # Check system status
make demo           # Run full demonstration
make simple-test    # Quick connectivity test
```

### Development
```bash
make format         # Format code
make lint           # Run linting
make test           # Run tests
make coverage       # Test coverage report
make clean          # Clean build artifacts
```

## 🔧 Configuration

### Environment Variables
Create `.env` file for configuration:
```env
# Logging
LOG_LEVEL=INFO

# Ports (optional - defaults provided)
COORDINATOR_PORT=50051
INTEGRATION_PORT=50200
REST_API_PORT=8000

# Integration credentials (optional)
SLACK_BOT_TOKEN=xoxb-your-token
GITHUB_TOKEN=ghp_your-token
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
JENKINS_URL=https://jenkins.company.com
JENKINS_USER=your-user
JENKINS_TOKEN=your-token
```

### System Ports
| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| REST API | 8000 | HTTP | External interface |
| Coordinator | 50051 | gRPC | Task management |
| Integration | 50200 | gRPC | External services |
| Manager Agent | 50100 | gRPC | Project management |
| Frontend Agent | 50101 | gRPC | UI development |
| Backend Agent | 50102 | gRPC | API development |
| Database Agent | 50103 | gRPC | Data management |
| DevOps Agent | 50104 | gRPC | Infrastructure |
| Integration Agent | 50105 | gRPC | Service integration |
| QA Agent | 50106 | gRPC | Quality assurance |

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate gRPC code
RUN python scripts/generate_grpc.py

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 50051 50200 50100-50106

# Start system
CMD ["python", "main.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  mcp-system:
    build: .
    ports:
      - "8000:8000"    # REST API
      - "50051:50051"  # Coordinator
      - "50200:50200"  # Integration
      - "50100-50106:50100-50106"  # Agents
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # Optional: Add Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Optional: Add PostgreSQL for persistence
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mcp
      POSTGRES_USER: mcp_user
      POSTGRES_PASSWORD: mcp_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

Build and run:
```bash
docker-compose up -d
```

## ☸️ Kubernetes Deployment

### Namespace
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mcp-system
```

### ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mcp-config
  namespace: mcp-system
data:
  LOG_LEVEL: "INFO"
  COORDINATOR_PORT: "50051"
  INTEGRATION_PORT: "50200"
  REST_API_PORT: "8000"
```

### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-distributed-system
  namespace: mcp-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-system
  template:
    metadata:
      labels:
        app: mcp-system
    spec:
      containers:
      - name: mcp-system
        image: mcp-distributed-system:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 50051
          name: coordinator
        - containerPort: 50200
          name: integration
        envFrom:
        - configMapRef:
            name: mcp-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-system-service
  namespace: mcp-system
spec:
  selector:
    app: mcp-system
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: coordinator
    port: 50051
    targetPort: 50051
  - name: integration
    port: 50200
    targetPort: 50200
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/
```

## 🔍 Monitoring & Health Checks

### Health Endpoints
```bash
# System health
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/v1/system/status

# Integration status
curl http://localhost:8000/api/v1/integrations
```

### Logs
```bash
# View logs
tail -f logs/mcp_system.log

# Docker logs
docker-compose logs -f mcp-system

# Kubernetes logs
kubectl logs -f deployment/mcp-distributed-system -n mcp-system
```

### Metrics
The system exposes metrics for monitoring:
- Active agents and their status
- Task counts by status
- Integration health
- Process resource usage

## 🛠️ Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :8000
lsof -i :50051

# Kill processes if needed
kill -9 <PID>
```

**Permission errors:**
```bash
# Fix file permissions
chmod +x main.py
chmod +x scripts/generate_grpc.py
```

**Missing dependencies:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**gRPC generation fails:**
```bash
# Manual gRPC generation
python -m grpc_tools.protoc \
  --proto_path=protos \
  --python_out=generated \
  --grpc_python_out=generated \
  protos/agent.proto
```

### Performance Tuning

**For high-load environments:**
```env
# Increase worker processes
MAX_WORKERS=20

# Increase timeouts
GRPC_TIMEOUT=30
HTTP_TIMEOUT=60

# Enable verbose logging for debugging
LOG_LEVEL=DEBUG
```

**Resource limits:**
- Minimum: 2GB RAM, 2 CPU cores
- Recommended: 4GB RAM, 4 CPU cores
- High-load: 8GB RAM, 8 CPU cores

### Scaling Considerations

**Horizontal scaling:**
- Each agent type can run multiple instances
- Use load balancer for REST API
- Coordinator can handle multiple agent instances

**Vertical scaling:**
- Increase `max_concurrent_tasks` per agent
- Add more worker threads to gRPC servers
- Increase memory limits

## 🔒 Security

### Production Security Checklist
- [ ] Change default ports if needed
- [ ] Use HTTPS for REST API (reverse proxy)
- [ ] Implement authentication for API endpoints
- [ ] Secure gRPC channels with TLS
- [ ] Rotate integration API keys regularly
- [ ] Use secrets management for credentials
- [ ] Enable firewall rules for internal ports
- [ ] Regular security updates

### Network Security
```bash
# Only expose REST API publicly
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
iptables -A INPUT -p tcp --dport 50051:50200 -s 127.0.0.1 -j ACCEPT
```

## 📊 Production Deployment

### Load Balancer Configuration (nginx)
```nginx
upstream mcp_api {
    server 127.0.0.1:8000;
    # Add more instances for HA
    # server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name api.yourcompany.com;

    location / {
        proxy_pass http://mcp_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /health {
        proxy_pass http://mcp_api/health;
        access_log off;
    }
}
```

### Systemd Service
```ini
[Unit]
Description=MCP Distributed System
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-system
ExecStart=/opt/mcp-system/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
```

Install and start:
```bash
sudo cp mcp-system.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mcp-system
sudo systemctl start mcp-system
```

---

## 🎯 Success Checklist

After deployment, verify:
- [ ] All services start without errors
- [ ] Health check returns "healthy" status
- [ ] All 7 agents are online and idle
- [ ] Integration services respond to health checks
- [ ] Can create and distribute tasks via API
- [ ] Integration operations work (Slack, Git, AWS, etc.)
- [ ] Process monitoring shows all services running
- [ ] API documentation accessible at `/docs`

**🎉 Your MCP Distributed System is now ready for production!**
