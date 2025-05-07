"""
FastAPI service for property amenity detection system.
Exposes endpoints to detect amenities from images and generate descriptions.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import hydra

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the PropertyAmenitySystem after Hydra initialization
# to avoid configuration issues
property_amenity_system = None

app = FastAPI(
    title="Property Amenity Detection API",
    description="API for detecting amenities in property images and generating descriptions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Initialize the PropertyAmenitySystem when the API starts."""
    global property_amenity_system
    
    # Initialize Hydra with default config
    config = hydra.compose(config_name="config", return_hydra_config=True)
    
    # Import here to avoid circular imports
    from core.amenity_system import PropertyAmenitySystem
    
    # Initialize the system
    property_amenity_system = PropertyAmenitySystem(config, logger=logger)
    logger.info("PropertyAmenitySystem initialized for API service")

@app.get("/")
def read_root():
    """Root endpoint to check if API is running."""
    return {"status": "active", "service": "Property Amenity Detection API"}

# Import and include the amenities router
from api.router import router as amenities_router
app.include_router(amenities_router)

if __name__ == "__main__":
    import uvicorn
    # When running directly, start the server
    uvicorn.run("api.service:app", host="0.0.0.0", port=8000, reload=True)