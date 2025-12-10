#!/bin/bash
# Quick start script for Face Recognition Web Application

echo "=================================="
echo "Face Recognition Web Application"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "Arcface.venv" ]; then
    echo "❌ Error: Virtual environment 'Arcface.venv' not found!"
    echo "Please create it first or run the notebook setup."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source Arcface.venv/bin/activate

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing FastAPI..."
    pip install fastapi uvicorn jinja2 python-multipart
fi

# Check if required directories exist
echo "📁 Checking project structure..."
if [ ! -d "Artifacts" ]; then
    echo "⚠️  Warning: Artifacts directory not found!"
    echo "Please run the ArcFace notebook first to generate the database."
fi

if [ ! -f "yolov8n-face.pt" ]; then
    echo "⚠️  Warning: YOLO model file not found!"
    echo "The model will need to be downloaded on first run."
fi

# Create uploads directory if it doesn't exist
mkdir -p static/uploads

echo ""
echo "🚀 Starting FastAPI application..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Add project root to PYTHONPATH so backend module can be imported
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default port
PORT=8000

# Check if port is already in use
check_port() {
    if command -v lsof > /dev/null 2>&1; then
        lsof -i :$1 > /dev/null 2>&1
    elif command -v netstat > /dev/null 2>&1; then
        netstat -tuln | grep -q ":$1 "
    elif command -v ss > /dev/null 2>&1; then
        ss -tuln | grep -q ":$1 "
    else
        return 1  # Assume port is available if we can't check
    fi
}

# Find available port starting from 8000
if check_port $PORT; then
    echo "⚠️  Port $PORT is already in use."
    
    # Try to find the process and offer to kill it
    PID=$(lsof -ti :$PORT 2>/dev/null || netstat -tulpn 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 | head -1)
    
    if [ ! -z "$PID" ] && [ "$PID" != "-" ]; then
        echo "   Found process using port $PORT: PID $PID"
        echo "   Attempting to kill the process..."
        kill $PID 2>/dev/null
        sleep 2
        if check_port $PORT; then
            echo "   Process still running, trying force kill..."
            kill -9 $PID 2>/dev/null
            sleep 1
        fi
        echo "   ✅ Process terminated"
    fi
    
    # Check if port is still in use
    if check_port $PORT; then
        echo "   🔄 Looking for an available port..."
        for p in 8001 8002 8003 8004 8005; do
            if ! check_port $p; then
                PORT=$p
                echo "   ✅ Found available port: $PORT"
                break
            fi
        done
        
        if [ $PORT = 8000 ]; then
            echo "   ❌ Could not find an available port (tried 8000-8005)"
            echo "   Please manually kill the process using port 8000 or choose a different port"
            exit 1
        fi
    else
        echo "   ✅ Port $PORT is now available"
    fi
fi

echo "📍 Access the web interface at: http://127.0.0.1:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the FastAPI application with uvicorn
# Use python -m uvicorn with PYTHONPATH set to ensure proper module resolution
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port $PORT

