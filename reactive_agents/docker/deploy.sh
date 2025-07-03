#!/bin/bash

# Reflex Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="reflex"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  up        Start the services"
    echo "  down      Stop the services"
    echo "  restart   Restart the services"
    echo "  logs      Show logs"
    echo "  status    Show service status"
    echo "  shell     Open shell in container"
    echo ""
    echo "Options:"
    echo "  --dev     Use development environment"
    echo "  --prod    Use production environment (default)"
    echo "  --build   Build images before starting"
    echo "  --pull    Pull latest images"
    echo "  --help    Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            ENVIRONMENT="development"
            COMPOSE_FILE="docker-compose.dev.yml"
            SERVICE_NAME="reflex-dev"
            shift
            ;;
        --prod)
            ENVIRONMENT="production"
            COMPOSE_FILE="docker-compose.yml"
            SERVICE_NAME="reflex"
            shift
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --pull)
            PULL_FLAG="--pull"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            COMMAND=$1
            shift
            ;;
    esac
done

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose is not installed.${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 Reflex Deployment Manager${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Compose file: ${COMPOSE_FILE}${NC}"

# Execute command
case $COMMAND in
    up)
        echo -e "${YELLOW}Starting Reflex services...${NC}"
        docker-compose -f "$COMPOSE_FILE" up -d $BUILD_FLAG $PULL_FLAG
        echo -e "${GREEN}✅ Services started successfully${NC}"
        echo -e "${BLUE}Run 'docker-compose -f ${COMPOSE_FILE} logs -f' to view logs${NC}"
        ;;
    down)
        echo -e "${YELLOW}Stopping Reflex services...${NC}"
        docker-compose -f "$COMPOSE_FILE" down
        echo -e "${GREEN}✅ Services stopped successfully${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting Reflex services...${NC}"
        docker-compose -f "$COMPOSE_FILE" restart
        echo -e "${GREEN}✅ Services restarted successfully${NC}"
        ;;
    logs)
        echo -e "${BLUE}Showing logs for Reflex services...${NC}"
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    status)
        echo -e "${BLUE}Service status:${NC}"
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    shell)
        echo -e "${BLUE}Opening shell in ${SERVICE_NAME} container...${NC}"
        docker-compose -f "$COMPOSE_FILE" exec "$SERVICE_NAME" bash
        ;;
    build)
        echo -e "${YELLOW}Building Reflex images...${NC}"
        ./build.sh
        echo -e "${GREEN}✅ Build completed${NC}"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
        show_usage
        exit 1
        ;;
esac 