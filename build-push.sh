#!/bin/bash

# Script to build and push Docker image to a registry

# Default image name and tag
IMAGE_NAME="multi-agent-rag"
TAG="latest"
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --registry)
      REGISTRY="$2/"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

FULL_IMAGE_NAME="${REGISTRY}${IMAGE_NAME}:${TAG}"

echo "Building Docker image: $FULL_IMAGE_NAME"
docker build -t "$FULL_IMAGE_NAME" .

if [ -n "$REGISTRY" ]; then
  echo "Pushing image to registry..."
  docker push "$FULL_IMAGE_NAME"
  echo "Image pushed successfully: $FULL_IMAGE_NAME"
fi

echo "Build complete!"
echo "To run this image: docker run -p 8501:8501 -e COHERE_API_KEY=your_key_here $FULL_IMAGE_NAME"