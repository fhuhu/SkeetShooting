DEFAULT_CONFIG = {
    "camera": {
        "index": 1,
    },
    "display": {
        "visible_by_default": False,
        "window_name": "Webcam - Detection d'oiseaux",
    },
    "captures": {
        "directory": "captures",
        "filename_format": "oiseau_{timestamp}.jpg",
        "timestamp_format": "%Y%m%d_%H%M%S",
    },
    "detection": {
        "confidence_threshold": 0.4,
        "capture_delay_seconds": 5,
    },
    "audio": {
        "enabled": True,
        "file": "alert.wav",
    },
    "model": {
        "weights": "yolov8n.pt",
    }
}
