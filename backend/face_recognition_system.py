"""
Face Recognition System Module
Extracts face recognition logic from the ArcFace notebook for use in Flask backend
"""
import os
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Face recognition specific imports
from ultralytics import YOLO
import insightface
from insightface.utils.face_align import norm_crop

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

class FaceRecognitionSystem:
    """Face Recognition System using YOLOv8-Face + InsightFace + ArcFace"""
    
    def __init__(self, base_dir=None):
        """
        Initialize the face recognition system
        
        Args:
            base_dir: Base directory path. If None, uses current directory.
        """
        if base_dir is None:
            self.BASE_DIR = Path(__file__).resolve().parent
        else:
            self.BASE_DIR = Path(base_dir)
        
        # Paths
        self.ARTIFACTS_DIR = self.BASE_DIR / "Artifacts"
        self.YOLO_MODEL_PATH = self.BASE_DIR / "yolov8n-face.pt"
        self.FRS_DATASET_DIR = self.BASE_DIR / "FRS_DATASET"
        self.STATIC_DIR = self.BASE_DIR / "static"
        self.REFERENCE_IMAGE_DIR = self.STATIC_DIR / "reference_faces"
        self.REFERENCE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.dataset_dirs = []
        for dataset_name in ["Small_Dataset", "FRS_DATASET", "persons"]:
            dataset_path = self.BASE_DIR / dataset_name
            if dataset_path.exists():
                self.dataset_dirs.append(dataset_path)
        self.reference_cache = {}
        
        # Initialize models
        self.face_detector = None
        self.face_model = None
        
        # Database
        self.X = None  # Embeddings matrix
        self.y = None  # Labels
        self.y_encoded = None  # Encoded labels
        self.label_encoder = None
        self.best_classifier = None
        self.person_to_index = {}
        self.index_to_person = {}
        self.RECOGNITION_THRESHOLD = 1.0480  # Default, will load from file
        
        # Load models
        self._load_models()
        
        # Load database
        self._load_database()
        self._initialize_reference_images()
    
    def _load_models(self):
        """Load YOLOv8-Face and InsightFace models"""
        print("🔄 Loading face detection and recognition models...")
        
        # Load YOLOv8-Face
        if not self.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.YOLO_MODEL_PATH}")
        self.face_detector = YOLO(str(self.YOLO_MODEL_PATH))
        
        # Load InsightFace
        self.face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0, det_size=(640, 640))
        
        print("✅ Models loaded successfully")
    
    def _load_database(self):
        """Load embeddings database and classifier"""
        from sklearn.preprocessing import LabelEncoder
        
        database_path = self.ARTIFACTS_DIR / "embedding_database.npy"
        labels_path = self.ARTIFACTS_DIR / "labels.npy"
        label_encoder_path = self.ARTIFACTS_DIR / "label_encoder.pkl"
        classifier_path = self.ARTIFACTS_DIR / "face_classifier.pkl"
        person_mapping_path = self.ARTIFACTS_DIR / "person_mapping.json"
        threshold_path = self.ARTIFACTS_DIR / "recognition_thresholds.json"
        
        if not database_path.exists():
            print("⚠️ Database not found. Please run the notebook first.")
            return
        
        # Load embeddings and labels
        self.X = np.load(database_path)
        self.y_encoded = np.load(labels_path)
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.y = self.label_encoder.inverse_transform(self.y_encoded)
        
        # Load person mappings
        if person_mapping_path.exists():
            with open(person_mapping_path, 'r') as f:
                mapping = json.load(f)
                self.person_to_index = mapping.get('person_to_index', {})
                self.index_to_person = {int(k): v for k, v in mapping.get('index_to_person', {}).items()}
        
        # Load classifier
        if classifier_path.exists():
            with open(classifier_path, 'rb') as f:
                self.best_classifier = pickle.load(f)
        
        # Load threshold
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
                self.RECOGNITION_THRESHOLD = threshold_data.get('recognition_threshold', 1.0480)
        
        print(f"✅ Database loaded: {self.X.shape[0]} embeddings, {len(self.person_to_index)} persons")

    def _initialize_reference_images(self):
        """Ensure each known person has a cached reference thumbnail"""
        if not self.person_to_index:
            return
        for person_name in self.person_to_index.keys():
            try:
                self._ensure_reference_image(person_name)
            except Exception:
                # Reference image generation is best-effort; continue on failure
                continue

    def _get_saved_reference_image(self, person_name):
        """Return existing saved reference image path if available"""
        person_dir = self.REFERENCE_IMAGE_DIR / person_name
        if not person_dir.exists():
            return None
        candidates = sorted(
            [p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if candidates:
            self.reference_cache[person_name] = candidates[0]
            return candidates[0]
        return None

    def _locate_dataset_image(self, person_name):
        """Find a source image for a person from known dataset directories"""
        for dataset_dir in self.dataset_dirs:
            person_dir = dataset_dir / person_name
            if not person_dir.exists():
                continue
            candidates = sorted(
                [p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            )
            if candidates:
                return candidates[0]
        return None

    def _create_thumbnail_from_path(self, person_name, source_path):
        """Create thumbnail from an on-disk image path"""
        try:
            with Image.open(source_path) as img:
                return self._save_reference_thumbnail(person_name, img)
        except Exception:
            return None

    def _save_reference_thumbnail(self, person_name, pil_image):
        """Save a resized thumbnail for quick UI display"""
        if pil_image is None:
            return None
        person_dir = self.REFERENCE_IMAGE_DIR / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        save_path = person_dir / "reference.jpg"
        if save_path.exists():
            return save_path

        image_copy = pil_image.copy().convert("RGB")
        image_copy.thumbnail((220, 220))
        image_copy.save(save_path, format="JPEG", quality=92)
        self.reference_cache[person_name] = save_path
        return save_path

    def _ensure_reference_image(self, person_name):
        """Ensure a reference image exists for the given person"""
        cached_path = self.reference_cache.get(person_name)
        if cached_path and cached_path.exists():
            return cached_path

        existing_path = self._get_saved_reference_image(person_name)
        if existing_path:
            return existing_path

        dataset_image = self._locate_dataset_image(person_name)
        if dataset_image:
            return self._create_thumbnail_from_path(person_name, dataset_image)

        return None

    def get_reference_image_url(self, person_name):
        """Return web-accessible reference image URL for a person"""
        path = self._ensure_reference_image(person_name)
        if not path or not self.STATIC_DIR.exists():
            return None
        try:
            rel_path = path.relative_to(self.STATIC_DIR)
        except ValueError:
            return None
        return f"/static/{rel_path.as_posix()}"

    def save_reference_image_from_pil(self, person_name, pil_image):
        """Persist a reference image provided as a PIL Image"""
        if pil_image is None:
            return None
        existing = self._get_saved_reference_image(person_name)
        if existing:
            return existing
        return self._save_reference_thumbnail(person_name, pil_image)

    def save_reference_image_from_array(self, person_name, img_array):
        """Persist a reference image from a numpy array (BGR)"""
        if img_array is None:
            return None
        existing = self._get_saved_reference_image(person_name)
        if existing:
            return existing
        try:
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            return self._save_reference_thumbnail(person_name, pil_image)
        except Exception:
            return None

    def save_reference_image_from_aligned_face(self, person_name, aligned_face_rgb):
        """Persist a reference image from an aligned RGB face crop"""
        if aligned_face_rgb is None:
            return None
        existing = self._get_saved_reference_image(person_name)
        if existing:
            return existing
        try:
            pil_image = Image.fromarray(aligned_face_rgb.astype(np.uint8))
            return self._save_reference_thumbnail(person_name, pil_image)
        except Exception:
            return None
    
    def load_image(self, image_path_or_array):
        """
        Load image from file path or numpy array
        
        Args:
            image_path_or_array: Image path (str) or numpy array
        
        Returns:
            numpy array: Image in BGR format
        """
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
            if img is None:
                # Try loading with PIL and converting
                pil_img = Image.open(image_path_or_array).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        elif isinstance(image_path_or_array, np.ndarray):
            # If already BGR, return as is
            if len(image_path_or_array.shape) == 3:
                return image_path_or_array.copy()
        else:
            # PIL Image
            img_array = np.array(image_path_or_array)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        return None
    
    def detect_faces_yolo(self, img, conf_threshold=0.4):
        """
        Detect faces using YOLOv8-Face
        
        Args:
            img: Image in BGR format
            conf_threshold: Confidence threshold
        
        Returns:
            List of detection results
        """
        results = self.face_detector(img, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf.item())
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        return detections
    
    def extract_landmarks_insightface(self, img, bbox, return_embedding=False):
        """
        Extract 5-point facial landmarks and optionally embedding
        
        Args:
            img: Full image in BGR format
            bbox: Bounding box [x1, y1, x2, y2]
            return_embedding: If True, also return embedding
        
        Returns:
            landmarks_5 or (landmarks_5, embedding) tuple
        """
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # First, try to detect on full image and match with YOLO bbox
        try:
            faces_full = self.face_model.get(img)
            
            if len(faces_full) > 0:
                # Find the face that best matches the YOLO bbox
                best_match = None
                best_match_score = float('inf')
                
                for face in faces_full:
                    face_bbox = face.bbox
                    face_center_x = (face_bbox[0] + face_bbox[2]) / 2
                    face_center_y = (face_bbox[1] + face_bbox[3]) / 2
                    
                    # Calculate distance between centers
                    center_distance = np.sqrt((bbox_center_x - face_center_x)**2 + (bbox_center_y - face_center_y)**2)
                    
                    # Normalize by bbox size
                    normalized_distance = center_distance / max(bbox_width, bbox_height)
                    
                    if normalized_distance < best_match_score and normalized_distance < 0.5:  # Within 50% of bbox size
                        best_match_score = normalized_distance
                        best_match = face
                
                if best_match is not None:
                    landmarks_5 = best_match.kps.copy()
                    
                    if return_embedding:
                        embedding = best_match.embedding.copy()
                        # Normalize embedding
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        else:
                            embedding = None
                        return landmarks_5, embedding
                    
                    return landmarks_5
        except Exception:
            pass  # Fall back to crop method
        
        # Fallback: Crop face region with padding and try again
        padding = 0.3  # Increased padding
        h, w = img.shape[:2]
        
        crop_x1 = max(0, int(x1 - bbox_width * padding))
        crop_y1 = max(0, int(y1 - bbox_height * padding))
        crop_x2 = min(w, int(x2 + bbox_width * padding))
        crop_y2 = min(h, int(y2 + bbox_height * padding))
        
        face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return (None, None) if return_embedding else None
        
        try:
            # Get landmarks from InsightFace on crop
            faces = self.face_model.get(face_crop)
            
            if len(faces) == 0:
                return (None, None) if return_embedding else None
            
            # Get the best face
            best_face = max(faces, key=lambda f: f.det_score)
            
            # Extract 5-point landmarks
            landmarks_5 = best_face.kps.copy()
            
            # Adjust landmarks to original image coordinates
            landmarks_5[:, 0] += crop_x1
            landmarks_5[:, 1] += crop_y1
            
            if return_embedding:
                embedding = best_face.embedding.copy()
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    embedding = None
                return landmarks_5, embedding
            
            return landmarks_5
        except Exception:
            return (None, None) if return_embedding else None
    
    def align_face(self, img, landmarks_5, output_size=112):
        """
        Align face using 5-point landmarks
        
        Args:
            img: Image in BGR format
            landmarks_5: 5 keypoints
            output_size: Size of aligned face
        
        Returns:
            Aligned face in RGB format
        """
        if landmarks_5 is None:
            return None
        
        try:
            if isinstance(output_size, tuple):
                output_size = output_size[0]
            
            aligned = norm_crop(img, landmarks_5, image_size=output_size)
            return aligned  # norm_crop returns RGB
        except Exception as e:
            print(f"Error in face alignment: {e}")
            return None
    
    def extract_arcface_embedding(self, aligned_face_rgb):
        """
        Extract ArcFace embedding from aligned face
        
        Args:
            aligned_face_rgb: Aligned face in RGB format
        
        Returns:
            512-dimensional embedding or None
        """
        try:
            aligned_bgr = cv2.cvtColor(aligned_face_rgb, cv2.COLOR_RGB2BGR)
            faces = self.face_model.get(aligned_bgr)
            
            if len(faces) > 0:
                best_face = max(faces, key=lambda f: f.det_score)
                embedding = best_face.embedding
            else:
                return None
            
            # Normalize embedding
            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                return None
            
            return embedding
        except Exception as e:
            return None
    
    def angular_distance(self, embedding1, embedding2):
        """
        Compute angular distance between two normalized embeddings
        
        Args:
            embedding1, embedding2: Normalized embedding vectors
        
        Returns:
            Angular distance in radians
        """
        cosine_sim = np.dot(embedding1, embedding2)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angular_dist = np.arccos(cosine_sim)
        return angular_dist
    
    def recognize_face(self, image_path_or_array, return_details=False):
        """
        Complete face recognition pipeline
        
        Args:
            image_path_or_array: Image path, numpy array, or PIL Image
            return_details: If True, return detailed information
        
        Returns:
            dict: Recognition results
        """
        if self.X is None or self.best_classifier is None:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': 0.0,
                'bbox': None,
                'is_unknown': True,
                'error': 'Database not loaded. Please run the notebook first.'
            }
        
        # Load image
        img = self.load_image(image_path_or_array)
        if img is None:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': 0.0,
                'bbox': None,
                'is_unknown': True,
                'error': 'Failed to load image'
            }
        
        # Detect faces
        detections = self.detect_faces_yolo(img, conf_threshold=0.4)
        
        if len(detections) == 0:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': 0.0,
                'bbox': None,
                'is_unknown': True,
                'error': 'No faces detected'
            }
        
        # Process the most confident detection
        best_detection = max(detections, key=lambda x: x['confidence'])
        bbox = best_detection['bbox']
        detection_confidence = best_detection['confidence']
        
        # Extract landmarks
        landmarks = self.extract_landmarks_insightface(img, bbox)
        if landmarks is None:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist(),
                'is_unknown': True,
                'error': 'Landmark extraction failed'
            }
        
        # Align face
        aligned_face = self.align_face(img, landmarks)
        if aligned_face is None:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist(),
                'is_unknown': True,
                'error': 'Face alignment failed'
            }
        
        # Extract embedding with fallback
        embedding = None
        embedding = self.extract_arcface_embedding(aligned_face)
        if embedding is None:
            # Fallback: Extract embedding during landmark extraction
            try:
                _, embedding = self.extract_landmarks_insightface(img, bbox, return_embedding=True)
            except:
                pass
        
        if embedding is None:
            return {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': None,
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist(),
                'is_unknown': True,
                'error': 'Embedding extraction failed'
            }
        
        # Compute angular distances
        angular_distances = []
        for db_embedding in self.X:
            dist = self.angular_distance(embedding, db_embedding)
            angular_distances.append(dist)
        
        angular_distances = np.array(angular_distances)
        
        # Find closest match
        min_distance_idx = np.argmin(angular_distances)
        min_distance = angular_distances[min_distance_idx]
        
        # Classify using threshold
        if min_distance <= self.RECOGNITION_THRESHOLD:
            # Known person
            predicted_label_idx = self.y_encoded[min_distance_idx]
            predicted_person = self.label_encoder.inverse_transform([predicted_label_idx])[0]
            
            # Use classifier for verification
            classifier_pred_idx = self.best_classifier.predict([embedding])[0]
            classifier_pred = self.label_encoder.inverse_transform([classifier_pred_idx])[0]
            classifier_proba = self.best_classifier.predict_proba([embedding])[0][classifier_pred_idx]
            
            # Use classifier prediction if high confidence, otherwise use distance-based
            if classifier_proba > 0.7 and classifier_pred == predicted_person:
                identity = classifier_pred
                confidence = float(classifier_proba)
            else:
                identity = predicted_person
                confidence = float(1.0 - (min_distance / self.RECOGNITION_THRESHOLD))
            
            reference_url = self.get_reference_image_url(identity)
            if not reference_url:
                saved_path = self.save_reference_image_from_aligned_face(identity, aligned_face)
                if saved_path:
                    reference_url = self.get_reference_image_url(identity)

            result = {
                'identity': identity,
                'confidence': confidence,
                'angular_distance': float(min_distance),
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist(),
                'is_unknown': False,
                'reference_image_url': reference_url
            }
        else:
            # Unknown person
            result = {
                'identity': 'unknown',
                'confidence': 0.0,
                'angular_distance': float(min_distance),
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist(),
                'is_unknown': True,
                'reference_image_url': None
            }
        
        if return_details:
            result['embedding'] = embedding
            result['aligned_face'] = aligned_face
            result['landmarks'] = landmarks.tolist()
            result['all_distances'] = angular_distances.tolist()
        
        return result
    
    def recognize_all_faces(self, image_path_or_array):
        """
        Recognize all faces in an image
        
        Args:
            image_path_or_array: Image path, numpy array, or PIL Image
        
        Returns:
            dict: Recognition results for all faces
        """
        if self.X is None or self.best_classifier is None:
            return {
                'num_faces_detected': 0,
                'num_faces_recognized': 0,
                'faces': [],
                'error': 'Database not loaded. Please run the notebook first.'
            }
        
        # Load image
        img = self.load_image(image_path_or_array)
        if img is None:
            return {
                'num_faces_detected': 0,
                'num_faces_recognized': 0,
                'faces': [],
                'error': 'Failed to load image'
            }
        
        # Detect all faces
        detections = self.detect_faces_yolo(img, conf_threshold=0.4)
        
        if len(detections) == 0:
            return {
                'num_faces_detected': 0,
                'num_faces_recognized': 0,
                'faces': [],
                'error': None
            }
        
        # Sort detections by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        faces_results = []
        num_recognized = 0
        
        # Process each detected face
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            detection_confidence = detection['confidence']
            
            face_result = {
                'face_id': idx + 1,
                'detection_confidence': float(detection_confidence),
                'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                'is_recognized': False,
                'identity': 'unknown',
                'confidence': None,
                'error': None,
                'reference_image_url': None
            }
            
            try:
                # Extract landmarks
                landmarks = self.extract_landmarks_insightface(img, bbox)
                if landmarks is None:
                    face_result['error'] = 'Landmark extraction failed'
                    faces_results.append(face_result)
                    continue
                
                # Align face
                aligned_face = self.align_face(img, landmarks)
                if aligned_face is None:
                    face_result['error'] = 'Face alignment failed'
                    faces_results.append(face_result)
                    continue
                
                # Extract embedding with fallback
                embedding = self.extract_arcface_embedding(aligned_face)
                if embedding is None:
                    try:
                        _, embedding = self.extract_landmarks_insightface(img, bbox, return_embedding=True)
                    except:
                        pass
                
                if embedding is None or embedding.shape != (512,):
                    face_result['error'] = 'Embedding extraction failed'
                    faces_results.append(face_result)
                    continue
                
                # Check if database has any embeddings
                if self.X is None or self.X.shape[0] == 0:
                    # Database is empty, mark as unknown
                    face_result['is_recognized'] = False
                    face_result['identity'] = 'unknown'
                    faces_results.append(face_result)
                    continue
                
                # Compute angular distances
                angular_distances = []
                for db_embedding in self.X:
                    dist = self.angular_distance(embedding, db_embedding)
                    angular_distances.append(dist)
                
                angular_distances = np.array(angular_distances)
                
                # Find closest match
                min_distance_idx = np.argmin(angular_distances)
                min_distance = angular_distances[min_distance_idx]
                
                # Check if recognized
                if min_distance <= self.RECOGNITION_THRESHOLD:
                    # Known person
                    predicted_label_idx = self.y_encoded[min_distance_idx]
                    predicted_person = self.label_encoder.inverse_transform([predicted_label_idx])[0]
                    
                    # Use classifier for verification
                    classifier_pred_idx = self.best_classifier.predict([embedding])[0]
                    classifier_pred = self.label_encoder.inverse_transform([classifier_pred_idx])[0]
                    classifier_proba = self.best_classifier.predict_proba([embedding])[0][classifier_pred_idx]
                    
                    # Use classifier prediction if high confidence, otherwise use distance-based
                    if classifier_proba > 0.7 and classifier_pred == predicted_person:
                        identity = classifier_pred
                        confidence = float(classifier_proba)
                    else:
                        identity = predicted_person
                        confidence = float(1.0 - (min_distance / self.RECOGNITION_THRESHOLD))
                    
                    reference_url = self.get_reference_image_url(identity)
                    if not reference_url:
                        saved_path = self.save_reference_image_from_aligned_face(identity, aligned_face)
                        if saved_path:
                            reference_url = self.get_reference_image_url(identity)

                    face_result['is_recognized'] = True
                    face_result['identity'] = identity
                    face_result['confidence'] = float(confidence)
                    face_result['reference_image_url'] = reference_url
                    num_recognized += 1
                else:
                    # Unknown person
                    face_result['is_recognized'] = False
                    face_result['identity'] = 'unknown'
                    
            except Exception as e:
                face_result['error'] = f'Processing error: {str(e)}'
            
            faces_results.append(face_result)
        
        return {
            'num_faces_detected': len(detections),
            'num_faces_recognized': num_recognized,
            'faces': faces_results,
            'error': None
        }
    
    def register_new_person(self, person_name, image_paths_or_arrays, min_images=3):
        """
        Register a new person by processing multiple images and adding to database
        
        Args:
            person_name: Name/ID of the person to register
            image_paths_or_arrays: List of image paths, PIL Images, or numpy arrays
            min_images: Minimum number of successful embeddings required
        
        Returns:
            dict: Registration results
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        
        print(f"Registering new person: {person_name}")
        
        if self.X is None or self.X.shape[0] == 0:
            # Initialize database if empty
            self.X = np.array([]).reshape(0, 512)
            self.y = np.array([])
            self.y_encoded = np.array([])
            self.label_encoder = LabelEncoder()
            # Initialize classifier if not already loaded
            if self.best_classifier is None:
                self.best_classifier = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        
        if person_name in self.person_to_index:
            return {'success': False, 'error': f'Person {person_name} already exists in database'}
        
        # Create embeddings directory for this person
        EMBEDDINGS_DIR = self.ARTIFACTS_DIR / "embeddings"
        person_emb_dir = EMBEDDINGS_DIR / person_name
        person_emb_dir.mkdir(parents=True, exist_ok=True)
        
        new_embeddings = []
        successful_count = 0
        failed_count = 0
        
        # Process each image
        for img_input in image_paths_or_arrays:
            try:
                # Load image
                img = self.load_image(img_input)
                if img is None:
                    failed_count += 1
                    continue

                # Store a reference image if we don't have one yet
                self.save_reference_image_from_array(person_name, img)
                
                # Detect faces
                detections = self.detect_faces_yolo(img, conf_threshold=0.4)
                if len(detections) == 0:
                    failed_count += 1
                    continue
                
                # Process best detection
                best_detection = max(detections, key=lambda x: x['confidence'])
                bbox = best_detection['bbox']
                
                # Extract landmarks AND embedding together
                try:
                    result = self.extract_landmarks_insightface(img, bbox, return_embedding=True)
                    if result is None:
                        failed_count += 1
                        continue
                    
                    landmarks, embedding_from_crop = result
                    if landmarks is None:
                        failed_count += 1
                        continue
                except Exception as e:
                    failed_count += 1
                    continue
                
                # Align face
                aligned_face = self.align_face(img, landmarks)
                if aligned_face is None:
                    failed_count += 1
                    continue
                
                # Try extracting embedding from aligned face
                embedding_from_aligned = self.extract_arcface_embedding(aligned_face)
                
                # Use embedding from aligned face if successful, otherwise use the one from crop
                if embedding_from_aligned is not None:
                    embedding = embedding_from_aligned
                elif embedding_from_crop is not None:
                    embedding = embedding_from_crop
                else:
                    failed_count += 1
                    continue
                
                # Verify embedding is valid
                if embedding is None or embedding.shape != (512,):
                    failed_count += 1
                    continue
                
                # Save embedding
                embedding_filename = f"embedding_{successful_count}.npy"
                embedding_path = person_emb_dir / embedding_filename
                np.save(embedding_path, embedding)
                
                new_embeddings.append(embedding)
                successful_count += 1
            except Exception as e:
                failed_count += 1
                continue
        
        # Check if we have enough embeddings
        if successful_count < min_images:
            return {
                'success': False,
                'error': f'Insufficient successful embeddings: {successful_count}/{min_images} required',
                'successful': successful_count,
                'failed': failed_count
            }
        
        # Add new embeddings to database
        if len(new_embeddings) == 0:
            return {
                'success': False,
                'error': 'No embeddings extracted from provided images',
                'successful': 0,
                'failed': failed_count
            }
        
        # Ensure embeddings array has correct shape
        new_embeddings_array = np.array(new_embeddings)
        if new_embeddings_array.ndim == 1:
            new_embeddings_array = new_embeddings_array.reshape(1, -1)
        
        # Verify embedding dimension
        if new_embeddings_array.shape[1] != 512:
            return {
                'success': False,
                'error': f'Invalid embedding dimension: {new_embeddings_array.shape[1]}, expected 512',
                'successful': successful_count,
                'failed': failed_count
            }
        
        # Update database
        if self.X.shape[0] == 0:
            self.X = new_embeddings_array
        else:
            self.X = np.vstack([self.X, new_embeddings_array])
        
        # Add labels
        new_labels = np.array([person_name] * successful_count)
        self.y = np.hstack([self.y, new_labels]) if self.y.size > 0 else new_labels
        
        # Update label encoder
        self.label_encoder.fit(self.y)
        self.y_encoded = self.label_encoder.transform(self.y)
        
        # Update person mappings
        if person_name not in self.person_to_index:
            idx = len(self.person_to_index)
            self.person_to_index[person_name] = idx
            self.index_to_person[idx] = person_name
        
        # Retrain classifier
        if self.best_classifier is None:
            self.best_classifier = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        
        self.best_classifier.fit(self.X, self.y_encoded)
        
        # Save updated database
        self._save_database()
        
        print(f"Person {person_name} registered successfully!")
        print(f"   Successful embeddings: {successful_count}")
        print(f"   Failed images: {failed_count}")
        
        return {
            'success': True,
            'person_name': person_name,
            'successful': successful_count,
            'failed': failed_count,
            'total_in_database': self.X.shape[0]
        }
    
    def _save_database(self):
        """Save the current database state"""
        from sklearn.preprocessing import LabelEncoder
        
        database_path = self.ARTIFACTS_DIR / "embedding_database.npy"
        labels_path = self.ARTIFACTS_DIR / "labels.npy"
        label_encoder_path = self.ARTIFACTS_DIR / "label_encoder.pkl"
        person_mapping_path = self.ARTIFACTS_DIR / "person_mapping.json"
        classifier_path = self.ARTIFACTS_DIR / "face_classifier.pkl"
        
        np.save(database_path, self.X)
        np.save(labels_path, self.y_encoded)
        
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(person_mapping_path, 'w') as f:
            json.dump({
                'person_to_index': self.person_to_index,
                'index_to_person': {str(k): v for k, v in self.index_to_person.items()}
            }, f, indent=2)
        
        if self.best_classifier is not None:
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.best_classifier, f)

# Global instance (will be initialized by Flask app)
face_system = None

def initialize_system(base_dir=None):
    """Initialize the global face recognition system"""
    global face_system
    face_system = FaceRecognitionSystem(base_dir)
    return face_system

def get_system():
    """Get the global face recognition system instance"""
    return face_system

