import numpy as np
from fastapi import Request
from backend.config import ALLOWED_EXTENSIONS

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

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
