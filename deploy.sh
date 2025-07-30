#!/bin/bash
# deploy.sh - Railway deployment script

echo "🚀 Preparing for Railway deployment..."

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
echo "🔑 Logging in to Railway..."
railway login

# Link or create project
echo "🔗 Linking to Railway project..."
railway link

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set PYTHONUNBUFFERED=1
railway variables set PYTHONDONTWRITEBYTECODE=1
railway variables set NIXPACKS_PYTHON_VERSION="3.11"

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment completed!"
echo "🌐 Your app should be available at the Railway-provided URL"
