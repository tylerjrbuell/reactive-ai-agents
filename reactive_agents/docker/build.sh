#!/bin/bash

# Reflex Docker Build Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="reflex"
VERSION="0.1.0a6"
REGISTRY="tylerbuell/reactive-agents"

echo -e "${BLUE}🚀 Building Reflex Docker Images${NC}"

# Function to build image
build_image() {
    local dockerfile=$1
    local tag=$2
    local context=${3:-../..}
    
    echo -e "${YELLOW}Building ${tag}...${NC}"
    docker build -f "$dockerfile" -t "$tag" "$context"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully built ${tag}${NC}"
    else
        echo -e "${RED}❌ Failed to build ${tag}${NC}"
        exit 1
    fi
}

# Build production image
echo -e "${BLUE}Building production image...${NC}"
build_image "Dockerfile" "${IMAGE_NAME}:${VERSION}"
build_image "Dockerfile" "${IMAGE_NAME}:latest"

# Build development image
echo -e "${BLUE}Building development image...${NC}"
build_image "Dockerfile.dev" "${IMAGE_NAME}:${VERSION}-dev"
build_image "Dockerfile.dev" "${IMAGE_NAME}:dev"

# Tag for registry if specified
if [ "$1" == "--registry" ]; then
    echo -e "${BLUE}Tagging images for registry...${NC}"
    
    docker tag "${IMAGE_NAME}:${VERSION}" "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${IMAGE_NAME}:latest"
    docker tag "${IMAGE_NAME}:${VERSION}-dev" "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev"
    docker tag "${IMAGE_NAME}:dev" "${REGISTRY}/${IMAGE_NAME}:dev"
    
    echo -e "${GREEN}✅ Images tagged for registry${NC}"
fi

# Push to registry if specified
if [ "$1" == "--push" ] || [ "$2" == "--push" ]; then
    echo -e "${BLUE}Pushing images to registry...${NC}"
    
    docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    docker push "${REGISTRY}/${IMAGE_NAME}:${VERSION}-dev"
    docker push "${REGISTRY}/${IMAGE_NAME}:dev"
    
    echo -e "${GREEN}✅ Images pushed to registry${NC}"
fi

echo -e "${GREEN}🎉 Build complete!${NC}"
echo -e "${BLUE}Available images:${NC}"
docker images | grep "$IMAGE_NAME" 