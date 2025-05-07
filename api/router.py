"""
API router for property amenity detection system.
Defines the endpoints and handlers for the FastAPI service.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
import io
import logging
import time
import uuid
from PIL import Image


# Setup logger
logger = logging.getLogger(__name__)

# Import after initialization to avoid circular dependencies
def get_amenity_system():
    """Get the initialized PropertyAmenitySystem instance."""
    from api.service import property_amenity_system
    if not property_amenity_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    return property_amenity_system

class AmenityResponse(BaseModel):
    """Response model for amenity detection API."""
    image_id: str
    amenities: dict
    description: str
    processing_time: float

# Create router
router = APIRouter(prefix="/api/amenities", tags=["amenities"])

@router.post("/detect", response_model=AmenityResponse)
async def detect_amenities(
    file: UploadFile = File(...),
    amenity_system = Depends(get_amenity_system)
):
    """
    Detect amenities in an uploaded property image.
    
    Args:
        file: The image file to analyze
        amenity_system: The PropertyAmenitySystem instance (injected)
        
    Returns:
        A JSON response with detected amenities and generated description
    """
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate a unique ID for this image
        image_id = str(uuid.uuid4())
        image_name = f"{image_id}_{file.filename}"
        
        # Process image
        start_time = time.time()
        amenities, description = amenity_system.process_image_from_memory(image, image_name)
        processing_time = time.time() - start_time
        
        # Return results
        return AmenityResponse(
            image_id=image_id,
            amenities=amenities,
            description=description,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/results")
def get_results(amenity_system = Depends(get_amenity_system)):
    """
    Get summary of all processed results.
    
    Args:
        amenity_system: The PropertyAmenitySystem instance (injected)
        
    Returns:
        A JSON response with summary of all processed images
    """
    try:
        # Get results as dictionary (convert dataframe to dict)
        results = amenity_system.get_all_results()
        results_dict = results.to_dict(orient="records")
        return {"results": results_dict}
    
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@router.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        A JSON response with status information
    """
    return {"status": "healthy", "service": "Property Amenity Detection API"}