import io
import os
import shutil
import tempfile
import zipfile
import traceback
from pathlib import Path
from PIL import Image

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.config import logger, face_system
from backend.utils import allowed_file, convert_numpy_types

router = APIRouter()

@router.post("/recognize")
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
    
    if face_system.face_detector is None or face_system.face_model is None:
        logger.error("Face recognition models not loaded")
        raise HTTPException(
            status_code=500,
            detail="Face recognition models not loaded. Please check server logs."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP)."
        )
    
    try:
        logger.info(f"Processing recognition request for file: {file.filename}")
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        result = face_system.recognize_all_faces(img)
        
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail=f'Unexpected result type from recognition: {type(result)}')
        
        if result.get('error'):
            raise HTTPException(status_code=500, detail=result['error'])
        
        response = {
            'success': True,
            'num_faces_detected': result.get('num_faces_detected', 0),
            'num_faces_recognized': result.get('num_faces_recognized', 0),
            'faces': []
        }
        
        faces_list = result.get('faces', [])
        for face in faces_list:
            if not isinstance(face, dict):
                continue
                
            try:
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
                continue
        
        response = convert_numpy_types(response)
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        logger.error(f"Exception in /recognize endpoint: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/register-person")
async def register_person(name: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Register a new person endpoint
    Accepts person name and multiple image files
    """
    if face_system is None:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    person_name = name.strip()
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Please upload at least one image")
    
    try:
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
            raise HTTPException(status_code=400, detail=result.get('error', 'Registration failed'))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error registering person: {str(e)}')


@router.post("/register-folder")
async def register_folder(folder: UploadFile = File(...)):
    """
    Register persons from a ZIP folder
    """
    logger.info(f"Received folder upload request: {folder.filename}")
    
    if face_system is None:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    if not folder.filename or not folder.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a ZIP file")
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'uploaded.zip')
        
        content = await folder.read()
        with open(zip_path, 'wb') as f:
            f.write(content)
        
        extract_dir = os.path.join(temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        persons_registered = 0
        total_images = 0
        errors = []
        
        extract_path = Path(extract_dir)
        person_folders = [d for d in extract_path.iterdir() if d.is_dir()]
        
        if len(person_folders) == 1:
            wrapper_folder = person_folders[0]
            subfolders = [d for d in wrapper_folder.iterdir() if d.is_dir()]
            if len(subfolders) > 0:
                person_folders = subfolders
            else:
                image_files_in_wrapper = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.gif', '*.webp']:
                    image_files_in_wrapper.extend(wrapper_folder.glob(ext))
                if len(image_files_in_wrapper) > 0:
                    person_folders = [wrapper_folder]
        
        if len(person_folders) == 0:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="ZIP file contains no valid folders.")
        
        for person_folder in person_folders:
            person_name = person_folder.name
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.gif', '*.webp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(person_folder.glob(ext))
                image_files.extend(person_folder.glob(ext.upper()))
            
            if len(image_files) == 0:
                errors.append(f"No images found in folder: {person_name}")
                continue
            
            images = []
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    face_system.save_reference_image_from_pil(person_name, img)
                    images.append(img)
                except Exception as e:
                    errors.append(f"Failed to load {img_path.name}: {str(e)}")
            
            if len(images) == 0:
                continue
            
            try:
                result = face_system.register_new_person(person_name, images, min_images=1)
                if result['success']:
                    persons_registered += 1
                    total_images += result['successful']
                else:
                    errors.append(f"Failed to register {person_name}: {result.get('error')}")
            except Exception as e:
                errors.append(f"Error registering {person_name}: {str(e)}")
        
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        if persons_registered == 0:
            raise HTTPException(status_code=400, detail=f"No persons registered. Errors: {'; '.join(errors[:3])}")
        
        response = {
            'success': True,
            'persons_registered': persons_registered,
            'total_images': total_images
        }
        if errors:
            response['warnings'] = errors[:10]
        return JSONResponse(content=response)
    
    except HTTPException:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    except Exception as e:
        logger.error("Error processing folder", exc_info=True)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def status():
    if face_system is None:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    return JSONResponse(content={
        'status': 'initialized',
        'num_embeddings': int(face_system.X.shape[0]) if face_system.X is not None else 0,
        'num_classes': len(face_system.person_to_index) if face_system.person_to_index else 0,
        'recognition_threshold': float(face_system.RECOGNITION_THRESHOLD) if hasattr(face_system, 'RECOGNITION_THRESHOLD') else 0.5
    })
