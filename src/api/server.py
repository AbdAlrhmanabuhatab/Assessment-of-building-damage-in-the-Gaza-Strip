import os
import time
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config.settings import config, BASE_DIR
from src.models.mobilenet import DamageClassifier
from src.xai.gradcam import GradCAMVisualizer

def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI web service."""
    app = FastAPI(
        title="Gaza Building Structural Damage AI",
        description=(
            "Production Deep Learning (MobileNetV2) and Explainable AI (Grad-CAM) "
            "service for rapid structural building damage classification and triage."
        ),
        version="3.3.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Dynamic Cache-Control headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # Initialize model manager (singleton in server process)
    classifier = DamageClassifier()

    def _sync_inference(contents: bytes) -> Dict[str, Any]:
        """Internal synchronous worker executed in threadpool."""
        result = classifier.predict(contents)
        class_idx = result["class_index"]
        rgb_vis = result["rgb_visualization"]
        tensor_inp = result["tensor_input"]

        # Compute Explainable AI Grad-CAM
        overlay_b64 = ""
        heatmap_b64 = ""
        try:
            heatmap = GradCAMVisualizer.generate_heatmap(
                img_array=tensor_inp,
                model=classifier.model,
                pred_index=class_idx
            )
            blended, colored_heat = GradCAMVisualizer.overlay_heatmap(rgb_vis, heatmap, alpha=0.55)
            overlay_b64 = GradCAMVisualizer.to_base64_data_uri(blended)
            heatmap_b64 = GradCAMVisualizer.to_base64_data_uri(colored_heat)
        except Exception as cam_err:
            print(f"[WARNING] Grad-CAM computation failed: {cam_err}")

        orig_b64 = GradCAMVisualizer.to_base64_data_uri(rgb_vis)

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "advisory": result["advisory"],
            "model_version": result["model_path"],
            "original_image_base64": orig_b64,
            "gradcam_overlay_base64": overlay_b64,
            "gradcam_heatmap_base64": heatmap_b64
        }

    # -------------------------------------------------------------
    # REST API ROUTES
    # -------------------------------------------------------------
    @app.get("/api/health", tags=["System"])
    def health_check():
        return {
            "status": "online",
            "model": os.path.basename(classifier.model_path),
            "preprocessing_mode": classifier.norm_mode,
            "classes": config.CLASS_NAMES
        }

    @app.get("/api/samples", tags=["Inference"])
    def get_preset_samples():
        ts = int(time.time())
        return [
            {
                "id": "demolish.jpg",
                "url": f"/static/samples/demolish.jpg?v={ts}",
                "title": "Critical Collapse (Demolish)",
                "category": "Demolish",
                "description": "Severely damaged multi-story structure with collapsed columns and rubble."
            },
            {
                "id": "repair_bank.jpg",
                "url": f"/static/samples/repair_bank.jpg?v={ts}",
                "title": "Moderate Damage (Repair)",
                "category": "Repair",
                "description": "Commercial building with damaged facade and awning, intact load-bearing structure."
            },
            {
                "id": "safe.jpg",
                "url": f"/static/samples/safe.jpg?v={ts}",
                "title": "Intact Structure (Safe)",
                "category": "Safe",
                "description": "No critical structural failure or collapse visible from this viewpoint."
            }
        ]

    @app.post("/api/predict", tags=["Inference"])
    async def predict_damage_endpoint(file: UploadFile = File(...)):
        # Security validation: File extension and MIME type
        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        has_image_ext = any(file.filename.lower().endswith(ext) for ext in valid_exts) if file.filename else False
        is_image_mime = file.content_type and file.content_type.startswith("image/")

        if not is_image_mime and not has_image_ext and file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file payload received.")

        # Security: Payload size protection
        if len(contents) > config.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File size exceeds maximum permitted limit (10MB).")

        try:
            # Offload heavy ML and Grad-CAM compute to async threadpool
            return await asyncio.to_thread(_sync_inference, contents)
        except (ValueError, FileNotFoundError) as client_err:
            raise HTTPException(status_code=400, detail=str(client_err))
        except Exception as server_err:
            raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(server_err)}")

    # -------------------------------------------------------------
    # STATIC ASSETS & FRONTEND HOSTING
    # -------------------------------------------------------------
    if config.STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

    @app.get("/", tags=["Frontend"])
    def serve_frontend():
        index_file = config.STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"service": "Gaza Building Damage AI API", "docs": "/docs"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
