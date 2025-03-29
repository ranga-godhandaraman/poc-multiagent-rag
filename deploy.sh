#!/bin/bash

# Simple deployment script for Multi-Agent RAG System

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "Please enter your Cohere API key:"
    read -r api_key
    echo "COHERE_API_KEY=$api_key" > .env
    echo ".env file created successfully."
else
    echo ".env file already exists."
fi

# Build and start the containers
echo "Building and starting containers..."
docker-compose up --build -d

echo "Deployment complete!"
echo "You can access the application at: http://localhost:8501"