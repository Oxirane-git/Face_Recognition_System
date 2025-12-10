"""
FastAPI Web Application for Face Recognition System
Uses YOLOv8-Face + InsightFace + ArcFace for face recognition
"""
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount
from pathlib import Path
from PIL import Image
import io
import sys
import zipfile
import shutil
import tempfile
import os
import logging
import traceback
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

from face_recognition_system import initialize_system, get_system

# Get project root directory (parent of backend folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Initialize FastAPI
app = FastAPI(title="FaceArt® Face Recognition API")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mount static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / 'static')), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(PROJECT_ROOT / 'templates'))

# Helper function for url_for in templates (FastAPI compatible)
def url_for_helper(request: Request, name: str, **path_params):
    """Helper function for url_for in templates (compatible with Flask syntax)"""
    if name == 'static':
        filename = path_params.get('filename', '')
        return f"/static/{filename}"
    elif name == 'serve_gif':
        filename = path_params.get('filename', '')
        return f"/gifs/{filename}"
    else:
        # For route names - use FastAPI's url_for
        try:
            return str(request.url_for(name, **path_params))
        except Exception:
            # Fallback to manual URL construction
            route_map = {
                'home': '/',
                'features': '/features',
                'try_now': '/try-now'
            }
            return route_map.get(name, '/')

# Configuration
UPLOAD_FOLDER = PROJECT_ROOT / 'static' / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Initialize face recognition system - BASE_DIR is project root
BASE_DIR = PROJECT_ROOT
print(f"🔄 Initializing face recognition system from: {BASE_DIR}")

try:
    face_system = initialize_system(base_dir=str(BASE_DIR))
    print("✅ Face recognition system initialized successfully")
except Exception as e:
    print(f"❌ Error initializing face recognition system: {e}")
    face_system = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================
# Routes
# ============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })


@app.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    """Render features page"""
    return templates.TemplateResponse("features.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })


@app.get("/try-now", response_class=HTMLResponse)
@app.get("/try_now", response_class=HTMLResponse)
async def try_now(request: Request):
    """Render try now page"""
    return templates.TemplateResponse("try_now.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    Face recognition endpoint
    Accepts image file and returns recognition results
    """
    if face_system is None:
        logger.error("Face recognition system is None - not initialized")
        raise HTTPException(
            status_code=500,
            detail="Face recognition system not initialized. Please check server logs."
        )
    
    # Check if models are loaded
    if face_system.face_detector is None or face_system.face_model is None:
        logger.error("Face recognition models not loaded")
        raise HTTPException(
            status_code=500,
            detail="Face recognition models not loaded. Please check server logs."
        )
    
    logger.info(f"Face system status: detector={face_system.face_detector is not None}, model={face_system.face_model is not None}, database_size={face_system.X.shape[0] if face_system.X is not None else 0}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)."
        )
    
    try:
        # Read image file
        logger.info(f"Processing recognition request for file: {file.filename}")
        image_bytes = await file.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Image loaded: {img.size}, mode: {img.mode}")
        
        # Perform recognition on all faces
        result = face_system.recognize_all_faces(img)
        logger.info(f"Recognition result type: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        
        # Check if result is a dict
        if not isinstance(result, dict):
            logger.error(f"Unexpected result type: {type(result)}")
            raise HTTPException(status_code=500, detail=f'Unexpected result type from recognition: {type(result)}')
        
        # Check for errors
        if result.get('error'):
            logger.error(f"Recognition error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Format response with all faces (use .get() with defaults to prevent KeyError)
        response = {
            'success': True,
            'num_faces_detected': result.get('num_faces_detected', 0),
            'num_faces_recognized': result.get('num_faces_recognized', 0),
            'faces': []
        }
        
        # Format each face result
        faces_list = result.get('faces', [])
        logger.info(f"Processing {len(faces_list)} face results")
        
        for face in faces_list:
            if not isinstance(face, dict):
                logger.warning(f"Skipping invalid face result: {type(face)}")
                continue
                
            try:
                # Convert numpy types to Python types for JSON serialization
                bbox = face.get('bbox', [])
                if bbox is not None:
                    bbox = convert_numpy_types(bbox)
                
                detection_conf = face.get('detection_confidence', 0.0)
                if detection_conf is not None:
                    detection_conf = float(detection_conf) * 100
                
                confidence = face.get('confidence')
                if confidence is not None:
                    confidence = round(float(confidence) * 100, 2)
                
                face_data = {
                    'face_id': int(face.get('face_id', 0)),
                    'detection_confidence': round(detection_conf, 2) if detection_conf is not None else 0.0,
                    'bbox': bbox if bbox is not None else [],
                    'is_recognized': bool(face.get('is_recognized', False)),
                    'identity': str(face.get('identity', 'unknown')),
                    'confidence': confidence,
                    'error': face.get('error'),
                    'reference_image_url': face.get('reference_image_url')
                }
                response['faces'].append(face_data)
            except Exception as face_error:
                logger.error(f"Error formatting face result: {str(face_error)}")
                logger.error(traceback.format_exc())
                # Continue with other faces even if one fails
                continue
        
        # Convert numpy types in response
        response = convert_numpy_types(response)
        
        logger.info(f"Returning response: {response['num_faces_detected']} faces, {response['num_faces_recognized']} recognized")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        logger.error(f"Exception in /recognize endpoint: {error_msg}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/register-person")
async def register_person(name: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Register a new person endpoint
    Accepts person name and multiple image files
    """
    if face_system is None:
        raise HTTPException(
            status_code=500,
            detail="Face recognition system not initialized"
        )
    
    person_name = name.strip()
    
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Please upload at least one image")
    
    try:
        # Convert uploaded files to PIL Images
        images = []
        for file in files:
            if file.filename and allowed_file(file.filename):
                image_bytes = await file.read()
                img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                try:
                    face_system.save_reference_image_from_pil(person_name, img)
                except Exception:
                    pass
                images.append(img)
        
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No valid image files uploaded")
        
        # Register person using face recognition system
        result = face_system.register_new_person(person_name, images, min_images=1)
        
        if result['success']:
            return JSONResponse(content={
                'success': True,
                'person_name': result['person_name'],
                'successful': result['successful'],
                'failed': result.get('failed', 0),
                'images_count': result['successful']
            })
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Registration failed')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error registering person: {str(e)}')


@app.post("/register-folder")
async def register_folder(folder: UploadFile = File(...)):
    """
    Register persons from a ZIP folder
    ZIP structure: Each subfolder is a person class, containing images
    """
    logger.info(f"Received folder upload request: {folder.filename}")
    
    if face_system is None:
        logger.error("Face recognition system not initialized")
        raise HTTPException(
            status_code=500,
            detail="Face recognition system not initialized"
        )
    
    if not folder.filename:
        logger.error("No filename provided")
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not folder.filename.endswith('.zip'):
        logger.error(f"Invalid file type: {folder.filename}")
        raise HTTPException(status_code=400, detail="Please upload a ZIP file")
    
    temp_dir = None
    try:
        logger.info(f"Processing ZIP file: {folder.filename}")
        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'uploaded.zip')
        
        # Save uploaded file
        logger.info("Reading ZIP file content...")
        content = await folder.read()
        logger.info(f"ZIP file size: {len(content)} bytes")
        
        with open(zip_path, 'wb') as f:
            f.write(content)
        
        # Extract ZIP file
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        logger.info("ZIP file extracted successfully")
        
        # Process each subfolder as a person class
        persons_registered = 0
        total_images = 0
        errors = []
        
        extract_path = Path(extract_dir)
        # List all items in extract directory
        all_items = list(extract_path.iterdir())
        logger.info(f"Found {len(all_items)} items in extracted directory")
        for item in all_items:
            logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Handle case where ZIP has a wrapper folder (e.g., Testfolder/person1/, Testfolder/person2/)
        # If there's only one folder and it contains subfolders, use that as the base
        person_folders = [d for d in extract_path.iterdir() if d.is_dir()]
        
        if len(person_folders) == 1:
            # Check if this single folder contains subfolders (person folders)
            wrapper_folder = person_folders[0]
            subfolders = [d for d in wrapper_folder.iterdir() if d.is_dir()]
            if len(subfolders) > 0:
                logger.info(f"Detected wrapper folder '{wrapper_folder.name}' containing {len(subfolders)} person folders")
                logger.info("Using subfolders as person folders")
                person_folders = subfolders
            # If no subfolders, check if it contains images directly (single person in wrapper)
            else:
                image_files_in_wrapper = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.gif', '*.webp']:
                    image_files_in_wrapper.extend(wrapper_folder.glob(ext))
                if len(image_files_in_wrapper) > 0:
                    logger.info(f"Wrapper folder '{wrapper_folder.name}' contains images directly, treating as single person")
                    # Use wrapper folder as person folder
                    person_folders = [wrapper_folder]
        
        logger.info(f"Found {len(person_folders)} person folders")
        
        if len(person_folders) == 0:
            # Check if there are any files at the root level
            root_files = [f for f in extract_path.iterdir() if f.is_file()]
            if root_files:
                error_detail = f"ZIP file contains files at root level but no folders. Found {len(root_files)} files. Each folder should represent a person class containing images."
            else:
                error_detail = "ZIP file does not contain any folders. Each folder should represent a person class."
            
            logger.error(error_detail)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=400,
                detail=error_detail
            )
        
        for person_folder in person_folders:
            person_name = person_folder.name
            logger.info(f"Processing person folder: {person_name}")
            
            # Get all images in this folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.gif', '*.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(person_folder.glob(ext))
                image_files.extend(person_folder.glob(ext.upper()))
            
            if len(image_files) == 0:
                error_msg = f"No images found in folder: {person_name}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
            
            logger.info(f"Found {len(image_files)} image files for {person_name}")
            
            # Load images
            images = []
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    try:
                        face_system.save_reference_image_from_pil(person_name, img)
                    except Exception as e:
                        logger.warning(f"Could not save reference image for {person_name}: {str(e)}")
                    images.append(img)
                except Exception as e:
                    error_msg = f"Failed to load image {img_path.name} in {person_name}: {str(e)}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            if len(images) == 0:
                error_msg = f"No valid images could be loaded from folder: {person_name}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
            
            # Register person
            try:
                logger.info(f"Registering {person_name} with {len(images)} images")
                result = face_system.register_new_person(person_name, images, min_images=1)
                if result['success']:
                    persons_registered += 1
                    total_images += result['successful']
                    logger.info(f"Successfully registered {person_name}: {result['successful']} embeddings")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    # If person already exists, note it but don't treat as critical error
                    if 'already exists' in error_msg.lower():
                        logger.info(f"Person {person_name} already exists, skipping")
                        errors.append(f"{person_name}: Already exists in database")
                    else:
                        logger.warning(f"Failed to register {person_name}: {error_msg}")
                        errors.append(f"Failed to register {person_name}: {error_msg}")
            except Exception as e:
                error_msg = f"Error registering {person_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
        
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logger.info(f"Registration complete. Registered: {persons_registered}, Total images: {total_images}, Errors: {len(errors)}")
        
        if persons_registered == 0:
            error_detail = 'No persons were successfully registered'
            if errors:
                # Include first few errors for context
                error_detail += f'. Errors: {"; ".join(errors[:3])}'
                if len(errors) > 3:
                    error_detail += f' (and {len(errors) - 3} more)'
            logger.error(f"Registration failed: {error_detail}")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=400,
                detail=error_detail
            )
        
        response = {
            'success': True,
            'persons_registered': persons_registered,
            'total_images': total_images
        }
        
        if errors:
            response['warnings'] = errors[:10]  # Limit warnings
        
        logger.info(f"Successfully returning response: {response}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return JSONResponse(content=response)
    
    except HTTPException as e:
        logger.error(f"HTTPException in register-folder: {e.detail}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in register-folder: {str(e)}", exc_info=True)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f'Error processing folder: {str(e)}')


@app.get("/status")
async def status():
    """Get system status"""
    if face_system is None:
        raise HTTPException(
            status_code=500,
            detail="Face recognition system not initialized"
        )
    
    return JSONResponse(content={
        'status': 'initialized',
        'num_embeddings': int(face_system.X.shape[0]) if face_system.X is not None else 0,
        'num_classes': len(face_system.person_to_index) if face_system.person_to_index else 0,
        'recognition_threshold': float(face_system.RECOGNITION_THRESHOLD) if hasattr(face_system, 'RECOGNITION_THRESHOLD') else 0.5
    })


@app.get("/static/uploads/{filename}")
async def uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/gifs/{filename}")
async def serve_gif(filename: str):
    """Serve GIF and video files from Gifs folder"""
    gifs_dir = BASE_DIR / 'Gifs'
    file_path = gifs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == '__main__':
    import uvicorn
    
    if face_system is None:
        print("⚠️ Warning: Face recognition system not initialized. Some features may not work.")
    
    print("\n" + "="*60)
    print("🚀 Starting Face Recognition Web Application")
    print("="*60)
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"🌐 Server will run on http://127.0.0.1:8000")
    print("="*60 + "\n")
    
    # Run app directly (app is already imported)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
