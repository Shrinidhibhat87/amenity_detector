"""
Streamlit app for property amenity detection system.
Provides a simple web interface to upload images and view detected amenities and descriptions.
"""
import streamlit as st
import requests
import pandas as pd
import time
import os
import sys
import logging

import torch
import os
# Fix the torch.classes path issue safely
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
except (AttributeError, TypeError):
    pass

from typing import Dict, Tuple
from PIL import Image

# Add the project root to Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Property Amenity Detection",
        page_icon="🏠",
        layout="wide",
    )
    
    st.title("Property Amenity Detection")
    st.markdown("""
    Upload a property image to detect amenities and generate a description.
    
    This app uses a vision-language model to analyze property images and identify amenities.
    """)
    
    # Sidebar for options
    with st.sidebar:
        st.header("Options")
        api_mode = st.radio(
            "Mode",
            ["Direct Inference"], # "Use API" is removed because it is not working yet
            help="Currently only direct inference is the working mode" #Choose whether to use the API or run inference directly
        )
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This application detects amenities in property images and generates descriptions.
        
        It uses a vision-language model to analyze the images and identify features.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("Detect Amenities"):
                with st.spinner("Processing image..."):
                    if api_mode == "Direct Inference":
                        # Use direct inference
                        amenities_by_room, description, detected_amenities, processing_time = process_image_directly(image)
                    else:
                        # Use the API
                        amenities_by_room, description, processing_time = process_image_via_api(uploaded_file)
                    
                    # Store results in session state for display
                    st.session_state.amenities_by_room = amenities_by_room
                    st.session_state.detected_amenities = detected_amenities
                    st.session_state.description = description
                    st.session_state.processing_time = processing_time
                    st.session_state.processed = True
    
    with col2:
        st.header("Results")
        
        if "processed" in st.session_state and st.session_state.processed:
            # Show processing time
            st.info(f"Processing Time: {st.session_state.processing_time:.2f} seconds")
            
            # Show description
            st.subheader("Property Description")
            st.write(st.session_state.description)
            
            # Show amenities
            # st.subheader("Detected Amenities")
            # Display a list of detected amenities (only the ones that are true)
            if st.session_state.detected_amenities:
                true_amenities = [
                    amenity.replace('_', ' ').title() 
                    for amenity, is_present in st.session_state.detected_amenities.items() 
                    if is_present
                ]
                
                if true_amenities:
                    st.markdown("### Detected amenities:")
                    for amenity in true_amenities:
                        st.markdown(f"- **{amenity}**")
                else:
                    st.write("No amenities detected")

            # Create a more user-friendly display of amenities
            #for room_type, amenities in st.session_state.amenities_by_room.items():
                #if any(amenities.values()):  # Only show room types with detected amenities
                    #with st.expander(f"{room_type.capitalize()}"):
                        #amenities_list = [amenity for amenity, is_present in amenities.items() if is_present]
                        #if amenities_list:
                            #st.write(", ".join(amenities_list))
                        #else:
                            #st.write("No amenities detected")

def process_image_via_api(uploaded_file) -> Tuple[Dict[str, Dict[str, bool]], str, float]:
    """
    Process an image via the FastAPI service.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        Tuple of (amenities, description, processing_time)
    """
    try:
        # Prepare the file for the API request
        files = {"file": uploaded_file.getvalue()}
        
        # Start timing
        start_time = time.time()
        
        # Make the API request
        response = requests.post(
            f"{API_URL}/api/amenities/detect",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        )
        
        # End timing (including network overhead)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return result["amenities"], result["description"], processing_time
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}, "Error processing image", processing_time
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}, f"Error: {str(e)}", 0

def process_image_directly(image) -> Tuple[Dict[str, Dict[str, bool]], str, float]:
    """
    Process an image directly using the PropertyAmenitySystem.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (amenities, description, processing_time)
    """
    try:
        import hydra
        # from hydra.core.singleton import Singleton

        # Check if Hydra is already initialized and clear it if needed
        # from hydra.core.singleton import Singleton
        # from hydra.core.config_store import ConfigStore
        from hydra.core.global_hydra import GlobalHydra
        # Clear Hydra if it's already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            
        # Also clear ConfigStore singleton if needed
        # ConfigStore.instance().clear()

        # Initialize Hydra with version_base to avoid deprecation warnings
        hydra.initialize(version_base=None, config_path="../config")
        
        # Compose the configuration, explicitly specifying the return_hydra_config option
        config = hydra.compose(config_name="config", return_hydra_config=True)
        
        # Import here to avoid circular imports
        from core.amenity_system import PropertyAmenitySystem


        # Initialize the system with configuration
        system = PropertyAmenitySystem(config, logger=logger)
        
        # Generate a temporary filename
        image_name = f"temp_{int(time.time())}.jpg"
        
        # Start timing
        start_time = time.time()
        
        # Process the image
        amenities_by_room, description, detected_amenities = system.process_image_from_memory(image, image_name)

        # End timing
        processing_time = time.time() - start_time
        
        return amenities_by_room, description, detected_amenities, processing_time
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {}, f"Error: {str(e)}", 0

def view_all_results():
    """View all processed results from the API."""
    try:
        response = requests.get(f"{API_URL}/api/amenities/results")
        
        if response.status_code == 200:
            results = response.json()["results"]
            
            if not results:
                st.info("No results available")
                return
                
            # Convert to DataFrame for better display
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="amenity_results.csv",
                mime="text/csv",
            )
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()