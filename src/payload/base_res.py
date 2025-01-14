from typing import List, Dict, Any

def create_response(status: str, data: List[Dict[str, Any]] = None, message: str = "") -> Dict[str, Any]:
    if data is None:
        data = []

    response = {
        "status": status,
        "message": message,
        "data": data,
    }

    return response