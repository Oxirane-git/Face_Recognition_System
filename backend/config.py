import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("face_recognition")

# Directories Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
UPLOAD_FOLDER = PROJECT_ROOT / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# System limits from ENV
PORT = int(os.environ.get("PORT", 8000))

# Initialize Face System
try:
    # Need to defer import to avoid sys.path circular issues if any exist
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from face_recognition_system import initialize_system
    
    face_system = initialize_system(base_dir=str(BASE_DIR))
    logger.info("✅ Face recognition system initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing face recognition system: {e}")
    face_system = None
