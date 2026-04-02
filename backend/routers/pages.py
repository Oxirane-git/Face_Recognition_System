from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from backend.config import PROJECT_ROOT, UPLOAD_FOLDER, BASE_DIR
from backend.utils import url_for_helper

router = APIRouter()
templates = Jinja2Templates(directory=str(PROJECT_ROOT / 'templates'))

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })

@router.get("/features", response_class=HTMLResponse)
async def features(request: Request):
    """Render features page"""
    return templates.TemplateResponse("features.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })

@router.get("/try-now", response_class=HTMLResponse)
@router.get("/try_now", response_class=HTMLResponse)
async def try_now(request: Request):
    """Render try now page"""
    return templates.TemplateResponse("try_now.html", {
        "request": request,
        "url_for": lambda name, **kwargs: url_for_helper(request, name, **kwargs)
    })

@router.get("/static/uploads/{filename}")
async def uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@router.get("/gifs/{filename}")
async def serve_gif(filename: str):
    """Serve GIF and video files from Gifs folder"""
    gifs_dir = BASE_DIR / 'Gifs'
    file_path = gifs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
