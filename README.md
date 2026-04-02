<div align="center">
  <img src="https://via.placeholder.com/150x150.png?text=FaceArt+Logo" alt="FaceArt Logo" width="150" height="150" />
  <h1>FaceArt® - Face Recognition System</h1>

  <p>
    <strong>A high-performance, dark-themed web interface for continuous and bulk face recognition integration.</strong>
  </p>

  <p>
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.11-blue.svg" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?logo=fastapi" />
    <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Face-yellow.svg" />
    <img alt="InsightFace" src="https://img.shields.io/badge/InsightFace-0.7.3-orange.svg" />
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg" />
  </p>
  
  <p>
    <a href="#features">Features</a> •
    <a href="#demo">Demo</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#api-endpoints">API</a>
  </p>
</div>

<hr/>

## ✨ Features

- **🚀 High-Precision Face Recognition**: Utilizes YOLOv8-Face for detection, InsightFace (buffalo_l) for landmarks, and ArcFace (r100) for embedding extraction and angular distance scoring.
- **👤 Dynamic Registration**: Register new individuals effortlessly by uploading their images. Processes embeddings in real-time.
- **🗂 Bulk Folder Support**: Upload a `.zip` archive containing directories of multiple profiles to process their embeddings in one go.
- **🌙 Premium Dark Theme UI**: Specifically crafted minimalistic dark theme matching the FaceArt® brand guidelines.
- **🔄 On-the-Fly Training**: System automatically retrains the SVM-RBF face classifier on database additions keeping detection state-of-the-art.

---

## 📸 Demo

*(Replace images below with your actual project screenshots or GIFs)*

| Face Recognition | Bulk Registration |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x250.png?text=Recognition+Demo" alt="Face Recognition Interface"> | <img src="https://via.placeholder.com/400x250.png?text=Bulk+Folder+Setup" alt="Bulk Registration"> |

---

## 🛠 Prerequisites

1. **Python 3.11+** installed.
2. Ensure you have the `yolov8n-face.pt` model inside the project root (*If missing, the system will auto-download an appropriate fallback or error out. Please ensure it is present for best results*).
3. **Database Initialization**: On your first boot, if the `Artifacts` array is missing it will create a fresh database upon your first person registration.

---

## 💻 Installation

1. **Clone & Environment Setup:**
   ```bash
   git clone <repository_url>
   cd Face_Recognition_System
   python -m venv Arcface.venv
   source Arcface.venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

*(Note: The build includes heavy ML libraries. For Dockerized deployment, see the Docker section).*

---

## 🚀 Running the Application

### Method 1: Using the Start Script (Recommended)
You can use the built-in startup script which automatically checks environment conditions, resolves ports, and starts the FastAPI server.
```bash
./run_web_app.sh
```

### Method 2: Manual Start with Uvicorn
```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Access the Web UI at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📚 Usage Guide

### 1. **Recognizing Faces**
- Navigate to the **Try Now** (`/try-now`) page.
- Upload an image utilizing Drag & Drop.
- The UI will present recognized identities, their respective confidence score, angular distance, and bounding box validation.

### 2. **Adding a Single Person**
- Navigate to the **Features** (`/features`) page.
- Under **Add New Person**, type an Identifiable Name (e.g., `Jane_Doe`).
- Upload 3-10 varied, clear images.
- System automatically builds 512-dimensional embeddings and pushes updates to `Artifacts`.

### 3. **Bulk Importing (ZIP Folder)**
- Format your `.zip` payload as follows:
  ```text
  dataset.zip
  ├── Jane_Doe/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── John_Smith/
      ├── photo1.png
      └── photo2.png
  ```
- Drop the zip file into **Add Complete Folder**. The classification engine will auto-process classes identically to their folder names.

---

## 📡 API Endpoints

The project exposes a fully usable API parallel to the UI interface. 

| Method | Endpoint | Description | Payloads |
|---|---|---|---|
| `GET` | `/status` | Server and Database integrity status | N/A |
| `POST` | `/recognize` | Analyzes image for face extraction | `file`: Image File Upload |
| `POST` | `/register-person` | Registers a single person profile | `name`: String, `files`: List of Images |
| `POST` | `/register-folder` | Processes a ZIP directory payload | `folder`: `.zip` File |

*(You can explore interactive API documentation by visiting `/docs` or `/redoc` when the server is running).*

---

## 🐳 Docker Support

To deploy within a containerized environment:

```bash
# Build the Docker image
docker build -t faceart-app .

# Run the container
docker run -p 8000:8000 faceart-app
```

---

## 🔍 Troubleshooting

- **"Face recognition system not initialized"**: Ensure InsightFace and YOLOv8 models exist. Also, verify `libGL` dependencies are met on Linux.
- **Port In Use Error**: The `run_web_app.sh` script automatically rectifies this, otherwise export another port via Uvicorn explicitly.
- **Zero Faces Detected**: The system enforces a default YOLO confidence of `0.4`. Ensure subjects are well-lit and not occluded.
- **Poor Recognition Accuracy**: Upload multiple angles per profile for better generalization within the SVM classifier.

---

<div align="center">
  <sub>Built with ❤️ using FastAPI, Ultralytics, and InsightFace. <a href="https://github.com/deepinsight/insightface">InsightFace Github</a>.</sub>
</div>
