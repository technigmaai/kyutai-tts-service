"""
Main entry point for Kyutai TTS Service
Coordinates all modules and starts the FastAPI server
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import modules
from config import SERVER_HOST, SERVER_PORT
from tts.engine import initialize_environment, initialize_tts_model
from audio.processing import initialize_zipenhancer
from api.routes import router

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Kyutai TTS API", 
    description="Generate downloadable speech from text.", 
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Include API routes
app.include_router(router)

def initialize_services():
    """Initialize all services and models"""
    logger.info("Initializing Kyutai TTS Service...")
    
    # Initialize environment
    initialize_environment()
    
    # Initialize TTS model
    tts_success = initialize_tts_model()
    if not tts_success:
        logger.error("Failed to initialize TTS model. Service cannot start.")
        return False
    
    # Initialize ZipEnhancer
    zipenhancer_success = initialize_zipenhancer()
    if zipenhancer_success:
        logger.info("ZipEnhancer initialized successfully.")
    else:
        logger.warning("ZipEnhancer initialization failed. Service will run without noise suppression.")
    
    logger.info("Service initialization complete.")
    return True

def main():
    """Main entry point"""
    # Initialize all services
    if not initialize_services():
        logger.error("Service initialization failed. Exiting.")
        return
    
    # Start the server
    logger.info(f"Launching FastAPI server on http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

if __name__ == "__main__":
    main() 