"""
FastAPI Web Application for Face Recognition System
Uses YOLOv8-Face + InsightFace + ArcFace for face recognition
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import PROJECT_ROOT, logger, face_system, PORT
from backend.routers import pages_router, api_router

# Initialize FastAPI
app = FastAPI(title="FaceArt® Face Recognition API")

# Mount static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / 'static')), name="static")

# Include Routers
app.include_router(pages_router)
app.include_router(api_router)

if __name__ == '__main__':
    import uvicorn

    if face_system is None:
        logger.warning("Face recognition system not initialized. Some features may not work.")

    logger.info("="*60)
    logger.info("🚀 Starting Face Recognition Web Application")
    logger.info("="*60)
    logger.info(f"📁 Base directory: {PROJECT_ROOT}")
    logger.info(f"🌐 Server will run on http://0.0.0.0:{PORT}")
    logger.info("="*60)

    # Run app directly
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
