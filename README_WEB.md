# FaceArt® - Face Recognition Web Application

A modern, dark-themed web interface for face recognition using YOLOv8-Face, InsightFace (buffalo_l), and ArcFace (r100). The application features a sleek dark UI and complete face recognition capabilities including person registration and folder-based bulk registration.

## Features

- **Face Recognition**: Upload an image and identify faces using the trained ArcFace model with angular distance scoring
- **Add New Person**: Register individuals by uploading multiple images - automatically processes embeddings and updates the database
- **Add Complete Folder**: Upload a ZIP file containing folder structures where each subfolder represents a person class
- **Dark Theme UI**: Modern, minimalist dark interface with FaceArt® branding
- **Real-time Processing**: Automatic database updates and classifier retraining after registration
- **Drag & Drop Support**: Easy file upload with drag-and-drop functionality

## Prerequisites

1. **Completed Notebook (Optional)**: The ArcFace notebook can be run first to generate initial database:
   - `Artifacts/embedding_database.npy`
   - `Artifacts/labels.npy`
   - `Artifacts/label_encoder.pkl`
   - `Artifacts/face_classifier.pkl`
   - `Artifacts/person_mapping.json`
   - `Artifacts/recognition_thresholds.json`
   
   **Note**: If the database doesn't exist, the system will automatically initialize it when you register your first person.

2. **Virtual Environment**: Use the `Arcface.venv` virtual environment

3. **YOLO Model**: `yolov8n-face.pt` should be in the project root (will be downloaded automatically if missing)

## Installation

1. **Activate Virtual Environment**:
   ```bash
   cd "/mnt/NewDisk/sahil_project/FRS copy"
   source Arcface.venv/bin/activate
   ```

2. **Install FastAPI and Dependencies**:
   ```bash
   pip install fastapi uvicorn jinja2 python-multipart
   ```
   
   Or install all from requirements:
   ```bash
   pip install -r requirements_flask.txt
   ```
   
   Note: The requirements file is named `requirements_flask.txt` for historical reasons but now contains FastAPI dependencies.

   All other dependencies (ultralytics, insightface, scikit-learn, etc.) should already be installed from the notebook setup.

## Running the Application

1. **Make sure you're in the project directory**:
   ```bash
   cd "/mnt/NewDisk/sahil_project/FRS copy"
   source Arcface.venv/bin/activate
   ```

2. **Run the FastAPI application**:
   
   Option 1: Use the run script (recommended):
   ```bash
   ./run_web_app.sh
   ```
   
   Option 2: Run directly with uvicorn (set PYTHONPATH first):
   ```bash
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
   ```
   
   Option 3: Run directly with uvicorn (alternative):
   ```bash
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Access the web interface**:
   - Open your browser and go to: `http://127.0.0.1:8000`
   - You'll see the FaceArt® home page with dark theme

## Pages Overview

### 🏠 Home Page
- Landing page with FaceArt® branding
- Quick access to "Upload Image" and "Clear" buttons
- Navigation to Features and Try Now pages

### ⚙️ Features Page
Two registration options:

1. **Add New Person**
   - Enter a unique person name/ID
   - Upload multiple images (3-10 recommended)
   - Automatically processes faces, extracts embeddings, and adds to database
   - Retrains the classifier automatically

2. **Add Complete Folder**
   - Upload a ZIP file containing folder structure
   - Each subfolder in the ZIP = one person class
   - Images inside each folder = training images for that person
   - Automatically processes all persons and creates separate classes

### 🔍 Try Now Page
- Upload an image for face recognition
- View recognition results with:
  - Identity (person name or "Unknown")
  - Confidence score
  - Angular distance
  - Detection confidence

## Usage Guide

### Recognizing Faces

1. Navigate to **Try Now** page (or click "Upload Image" from Home)
2. Click or drag-and-drop an image into the upload area
3. Click **Recognize Face** button
4. View the recognition results with confidence scores and metrics

### Adding a New Person (Individual Images)

1. Navigate to **Features** page
2. In the "Add New Person" card:
   - Enter a unique name/ID for the person (e.g., "John_Doe", "EMP001")
   - Upload 3-10 clear face images (drag-and-drop or click to select)
   - Click **Register Person**
3. The system will:
   - Detect faces in each image
   - Extract and align faces using YOLOv8-Face + InsightFace
   - Generate 512-dimensional ArcFace embeddings
   - Save embeddings to `Artifacts/embeddings/{person_name}/`
   - Update the master database
   - Retrain the classifier
   - Save all updated files automatically

### Adding Multiple Persons (ZIP Folder)

1. Prepare a ZIP file with this structure:
   ```
   dataset.zip
   ├── person1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── image3.jpg
   ├── person2/
   │   ├── photo1.png
   │   └── photo2.png
   └── person3/
       ├── img1.jpg
       └── img2.jpg
   ```

2. Navigate to **Features** page
3. In the "Add Complete Folder" card:
   - Click to select or drag-and-drop the ZIP file
   - Click **Process Folder**
4. The system will:
   - Extract the ZIP file
   - Process each folder as a separate person class
   - Register all persons automatically
   - Show summary of registered persons and total images

**Note**: Each subfolder name will be used as the person's identifier/class name.

## Project Structure

```
FRS copy/
├── backend/
│   ├── app.py                      # FastAPI web application
│   ├── face_recognition_system.py  # Face recognition module (YOLOv8 + InsightFace + ArcFace)
│   └── __pycache__/                # Python cache files
├── templates/
│   ├── index.html                  # Home page
│   ├── features.html               # Features page (add person/folder)
│   └── try_now.html                # Try Now page (recognition)
├── static/
│   ├── style.css                   # Dark theme styling
│   ├── script.js                   # Frontend JavaScript
│   └── uploads/                    # Temporary upload directory
├── Artifacts/                      # Generated artifacts (auto-created)
│   ├── embeddings/                 # Per-person embeddings
│   │   ├── person1/
│   │   │   ├── embedding_0.npy
│   │   │   └── ...
│   │   └── person2/
│   ├── embedding_database.npy      # Master embedding database
│   ├── labels.npy                  # Encoded labels
│   ├── label_encoder.pkl           # Label encoder
│   ├── face_classifier.pkl         # Trained classifier (SVM-RBF)
│   ├── person_mapping.json         # Person to index mapping
│   └── recognition_thresholds.json # Recognition thresholds
├── Notebook/
│   └── ArcFace.ipynb               # Training notebook (optional)
└── yolov8n-face.pt                 # YOLOv8-Face model
```

## API Endpoints

### `GET /`
Home page - Landing page with FaceArt® branding

### `GET /features`
Features page - Options to add new person or upload complete folder

### `GET /try-now` or `GET /try_now`
Try Now page - Face recognition interface

### `POST /recognize`
Face recognition endpoint
- **Input**: Image file (multipart/form-data, field: `file`)
- **Output**: JSON with recognition results
  ```json
  {
    "success": true,
    "result": {
      "identity": "person_name" or "unknown",
      "confidence": 85.23,
      "angular_distance": 0.4521,
      "detection_confidence": 92.45,
      "is_unknown": false,
      "bbox": [x1, y1, x2, y2]
    }
  }
  ```

### `POST /register-person`
Register a new person with individual images
- **Input**: 
  - `name` (form field): Person name/ID
  - `files` (multipart/form-data): Multiple image files
- **Output**: JSON with registration status
  ```json
  {
    "success": true,
    "person_name": "John_Doe",
    "successful": 5,
    "failed": 1,
    "images_count": 5
  }
  ```

### `POST /register-folder`
Register multiple persons from ZIP folder
- **Input**: 
  - `folder` (multipart/form-data): ZIP file containing folder structure
- **Output**: JSON with processing results
  ```json
  {
    "success": true,
    "persons_registered": 3,
    "total_images": 15,
    "warnings": [] // Optional, if any errors occurred
  }
  ```

### `GET /status`
Get system status
- **Output**: JSON with database status
  ```json
  {
    "status": "initialized",
    "num_embeddings": 150,
    "num_classes": 10,
    "recognition_threshold": 1.0480
  }
  ```

## Technical Details

### Face Recognition Pipeline

1. **Face Detection**: YOLOv8-Face detector (conf threshold: 0.4)
2. **Landmark Extraction**: InsightFace buffalo_l model (5-point landmarks)
3. **Face Alignment**: `norm_crop` utility for proper face alignment (112x112)
4. **Embedding Extraction**: ArcFace r100 model (512-dimensional embeddings)
5. **L2 Normalization**: All embeddings are normalized
6. **Distance Metric**: Angular distance (arccos of cosine similarity)
7. **Classification**: SVM-RBF classifier with probability estimation
8. **Threshold Calibration**: Automatic threshold for known vs. unknown faces

### Database Updates

When registering new persons:
- Embeddings are saved per-person in `Artifacts/embeddings/{person_name}/`
- Master database (`embedding_database.npy`) is updated
- Labels are re-encoded with updated label encoder
- Classifier is retrained on the complete dataset
- All artifacts are automatically saved

## Troubleshooting

### "Database not loaded" / Empty Database

- **Solution**: This is normal if you haven't run the notebook. Simply register your first person through the web interface - the database will be automatically initialized.

### "Face recognition system not initialized" Error

- Check that `yolov8n-face.pt` exists in the project root
- Verify InsightFace models can be downloaded (will download automatically on first run)
- Check the server console for detailed error messages
- Ensure all dependencies are installed in the virtual environment

### "No faces detected" Error

- Ensure uploaded images contain clear, visible faces
- Try images with better lighting
- Check that faces are not too small in the image
- YOLOv8-Face detection threshold is 0.4 - faces below this confidence won't be detected

### Port Already in Use

- Change the port in the uvicorn command:
  ```bash
  uvicorn backend.app:app --reload --host 127.0.0.1 --port 8001  # Change port
  ```

### ZIP Folder Not Processing

- Ensure ZIP file contains folders (not just loose images)
- Each folder should be named as the person identifier
- Check that images inside folders are valid (JPG, PNG, etc.)
- Maximum file size is 100MB

### Low Recognition Accuracy

- Ensure you register multiple images per person (3-10 recommended)
- Use clear, well-lit images with different angles
- Avoid blurry or heavily edited images
- Check recognition threshold in status endpoint

## Notes

- The web application provides complete face recognition functionality without needing the notebook
- Images are processed in real-time - no need to manually run notebook cells
- Database is automatically updated and saved after each registration
- The system supports both individual person registration and bulk folder-based registration
- All embeddings are stored in NPY format for efficient loading
- The classifier is automatically retrained after each new person registration

## Development

To run in development mode with auto-reload:
```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

For production deployment, use uvicorn with workers:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Performance Tips

- For faster recognition, ensure embeddings database is not too large
- Consider periodic database optimization by retraining with notebook
- Use appropriate image sizes (recommended: 640x480 to 1920x1080)
- Batch processing in folder upload may take time for large ZIP files

## License

This project uses open-source models:
- YOLOv8-Face (Ultralytics)
- InsightFace (DeepInsight)
- ArcFace (InsightFace)

## Support

For issues or questions:
1. Check the server console for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure the virtual environment is activated
4. Check that model files are in the correct locations
