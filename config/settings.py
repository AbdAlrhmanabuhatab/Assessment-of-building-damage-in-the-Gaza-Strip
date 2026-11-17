import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# Base Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

@dataclass(frozen=True)
class AppConfig:
    # Service Network Settings
    HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("APP_PORT", "8000"))
    DEBUG: bool = os.getenv("APP_DEBUG", "False").lower() in ("true", "1", "t")
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ])
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024

    # ML Pipeline & Geometry
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    IMAGE_CHANNELS: int = 3
    NUM_CLASSES: int = 3
    CLASS_NAMES: Tuple[str, ...] = ("Safe", "Repair", "Demolish")
    CLASS_LABELS: Dict[str, int] = field(default_factory=lambda: {
        "safe": 0,
        "repair": 1,
        "demolish": 2
    })

    # Dataset Paths
    RAW_DATASET_DIR: Path = BASE_DIR / "dataset"
    SPLIT_DATASET_DIR: Path = BASE_DIR / "dataset_split"
    STATIC_DIR: Path = BASE_DIR / "static"
    WEIGHTS_DIR: Path = BASE_DIR / "weights"

    # Model Checkpoint Candidates (Ordered by priority)
    MODEL_CANDIDATES: Tuple[Tuple[str, str], ...] = (
        ("weights/building_damage_model_v2.keras", "mobilenet"),
        ("building_damage_model_v2.keras", "mobilenet"),
        ("weights/building_damage_model_finetuned.h5", "div255"),
        ("building_damage_model_finetuned.h5", "div255"),
        ("weights/building_damage_model.h5", "div255"),
        ("building_damage_model.h5", "div255"),
    )

    # Standard Civil Engineering Safety Protocols
    ENGINEERING_ADVISORIES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Demolish": {
            "status_badge": "CRITICAL HAZARD - DEMOLISH",
            "severity_level": "High (Catastrophic)",
            "action_summary": "Immediate structural exclusion zone required. Structure is compromised beyond safe repair.",
            "protocols": [
                "Evacuate all occupants and establish a perimeter exclusion buffer (1.5x building height).",
                "Disconnect remaining municipal gas, electricity, and water mains immediately.",
                "Prohibit any entry under slabs or cantilevered balconies.",
                "Controlled demolition authorization recommended by licensed municipal structural engineer."
            ],
            "color_theme": "danger"
        },
        "Repair": {
            "status_badge": "STRUCTURAL REPAIR REQUIRED",
            "severity_level": "Moderate (Compromised)",
            "action_summary": "Primary structural elements partially intact. Comprehensive rehabilitation required before re-occupancy.",
            "protocols": [
                "Temporarily restrict occupancy until temporary shoring and propping are installed.",
                "Conduct ultrasonic or rebound hammer non-destructive testing on columns and beams.",
                "Remove loose masonry, unreinforced parapets, and hazardous overhanging debris.",
                "Formulate retrofitting plan (carbon fiber wrapping, epoxy injection, or steel jacketing)."
            ],
            "color_theme": "warning"
        },
        "Safe": {
            "status_badge": "APPARENTLY INTACT - LOW HAZARD",
            "severity_level": "Low (Surface / Superficial)",
            "action_summary": "No critical load-bearing collapse observed from this exterior perspective.",
            "protocols": [
                "Verify secondary views and inspect foundation settlement or subterranean utility leaks.",
                "Inspect interior shear walls for hairline settlement cracks not visible externally.",
                "Safe for conditional shelter or logistical access pending full interior clearance."
            ],
            "color_theme": "success"
        }
    })

config = AppConfig()
