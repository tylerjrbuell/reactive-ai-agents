# Reflex Docker Setup

This directory contains the official Docker configuration for the Reflex reactive agents framework.

## 🚀 Quick Start

### Production Deployment

```bash
# Build and start all services
./deploy.sh up --build

# Or using docker-compose directly
docker-compose up -d --build
```

### Development Environment

```bash
# Start development environment
./deploy.sh --dev up --build

# Or using docker-compose directly
docker-compose -f docker-compose.dev.yml up -d --build
```

## 📁 File Structure

```
docker/
├── Dockerfile              # Production image
├── Dockerfile.dev          # Development image
├── docker-compose.yml      # Production compose file
├── docker-compose.dev.yml  # Development compose file
├── .dockerignore           # Docker ignore file
├── build.sh                # Build script
├── deploy.sh               # Deployment script
├── config/                 # Configuration files
│   └── reflex.env          # Environment variables
└── README.md               # This file
```

## 🐳 Docker Images

### Production Image (`reflex:latest`)

- Based on Python 3.11 slim
- Optimized for production use
- Includes only runtime dependencies
- Non-root user for security
- Health checks enabled

### Development Image (`reflex:dev`)

- Based on Python 3.11 slim
- Includes development tools
- Debugging capabilities
- Hot reloading support
- Additional development packages

## 🛠️ Build Scripts

### Building Images

```bash
# Build all images
./build.sh

# Build with registry tagging
./build.sh --registry

# Build and push to registry
./build.sh --registry --push
```

### Manual Building

```bash
# Production image
docker build -f Dockerfile -t reflex:latest ../..

# Development image
docker build -f Dockerfile.dev -t reflex:dev ../..
```

## 🚀 Deployment

### Using the Deployment Script

```bash
# Production deployment
./deploy.sh up              # Start services
./deploy.sh down            # Stop services
./deploy.sh restart         # Restart services
./deploy.sh logs            # View logs
./deploy.sh status          # Check status
./deploy.sh shell           # Open shell

# Development deployment
./deploy.sh --dev up        # Start dev environment
./deploy.sh --dev shell     # Open dev shell
```

### Manual Docker Compose

```bash
# Production
docker-compose up -d
docker-compose down
docker-compose logs -f

# Development
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml exec reflex-dev bash
```

## 🔧 Configuration

### Environment Variables

Copy and customize the configuration file:

```bash
cp config/reflex.env config/.env
# Edit config/.env with your settings
```

Key configuration options:

```env
# Model Configuration
DEFAULT_MODEL=ollama:qwen2:7b
OLLAMA_HOST=http://ollama:11434

# API Keys
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Vector Database
CHROMA_HOST=http://chromadb:8000

# Agent Settings
MAX_ITERATIONS=10
CONTEXT_WINDOW=8000
MEMORY_ENABLED=true
```

### Volume Mounts

The compose files create persistent volumes for:

- `/app/data` - Agent data and storage
- `/app/logs` - Application logs
- `/app/config` - Configuration files
- Ollama models and ChromaDB data

## 🌐 Services

### Core Services

1. **Reflex Agent** (`reflex`)

   - Main application container
   - Ports: 8000
   - Health checks enabled

2. **Ollama** (`ollama`)

   - Local LLM inference server
   - Ports: 11434
   - Persistent model storage

3. **ChromaDB** (`chromadb`)
   - Vector database for memory
   - Ports: 8001
   - Persistent data storage

### Development Services

Additional services in development:

4. **Redis** (`redis-dev`)
   - Caching and session storage
   - Ports: 6379

## 🧪 Usage Examples

### Running an Agent

```bash
# Start services
./deploy.sh up

# Run an agent task
docker-compose exec reflex reflex make agent --task "What is the weather today?"

# Interactive mode
docker-compose exec reflex reflex make agent --interactive
```

### Development Workflow

```bash
# Start development environment
./deploy.sh --dev up

# Open development shell
./deploy.sh --dev shell

# Inside container:
reflex make agent --task "Test task"
pytest
black .
```

### Custom Model Setup

```bash
# Pull a model in Ollama
docker-compose exec ollama ollama pull qwen2:7b

# Use the model
docker-compose exec reflex reflex make agent --model "ollama:qwen2:7b" --task "Hello"
```

## 🔍 Monitoring and Debugging

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f reflex
docker-compose logs -f ollama
```

### Health Checks

```bash
# Check service health
docker-compose ps

# Manual health check
docker-compose exec reflex reflex version
```

### Debugging

Development image includes debugging tools:

```bash
# Start with debugger
docker-compose -f docker-compose.dev.yml exec reflex-dev python -m debugpy --listen 0.0.0.0:5678 -m reactive_agents.console.cli

# Connect debugger to localhost:5678
```

## 🔒 Security

### Production Security

- Non-root user (`reflex`)
- Read-only configuration mounts
- Network isolation
- Health checks
- Resource limits (can be added)

### Environment Security

```bash
# Set sensitive environment variables
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Or use .env file (not committed to git)
echo "GROQ_API_KEY=your-key" > config/.env
```

## 📊 Performance Tuning

### Resource Limits

Add to docker-compose.yml:

```yaml
services:
  reflex:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
```

### Scaling

```bash
# Scale reflex service
docker-compose up -d --scale reflex=3
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Permission issues**: Check volume permissions
3. **Memory issues**: Increase Docker memory limits
4. **Model loading**: Check Ollama service logs

### Debug Commands

```bash
# Check container status
docker-compose ps

# Inspect container
docker-compose exec reflex bash

# Check resource usage
docker stats

# View system info
docker system info
```

## 🔄 Updates and Maintenance

### Updating Images

```bash
# Pull latest images
docker-compose pull

# Rebuild with latest code
./deploy.sh up --build

# Clean up old images
docker image prune -f
```

### Backup and Restore

```bash
# Backup volumes
docker run --rm -v reflex_data:/data -v $(pwd):/backup alpine tar czf /backup/reflex_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v reflex_data:/data -v $(pwd):/backup alpine tar xzf /backup/reflex_backup.tar.gz -C /data
```

## 📚 Additional Resources

- [Reflex Documentation](../../README.md)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ollama Documentation](https://ollama.ai/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
