# Amenity Detection and Description System
This system automatically indentifies and detects amenities in property images. Additionally the system also generates a natural language description of the image. It uses a vision-language model, currently we have ([Llava](https://llava-vl.github.io/)), but this can easily be extended further.

## Project Scope

### Amenity Detection
The system is designed to detect/identify a comprehensive range of amenities in an image. It is primarily organised by room type, although this is still not a perfect solution. Currently the organisation looks like:

1. **Kitchen**: refrigerator, oven, microwave, etc.
2. **Living Room**: sofa, TV, coffee table, etc.
3. **Bedroom**: bed, wardrobe, desk, etc.
4. **Bathroom**: toilet, shower, sink, etc.
5. **Outdoor**: patio, pool, garden, etc.
6. **Common**: wifi, heating, security features, etc.

One can refer to core/amenity_schema.py which has the **AMENITY_SCHEMA** object detailing out the individual amenities.

### Technical Overview
The task is to detect, store and then generate a natural language description.

1. **Visual Recognition/Detection**: Use of ([Llava](https://huggingface.co/docs/transformers/en/model_doc/llava)) model from hugging face.
2. **Data Storage**: Store whatever results the detection algorithm comes out with in both CSV and SQLite format for easy access and future analysis.
3. **Description Generation**: Use of the same ([Llava](https://huggingface.co/docs/transformers/en/model_doc/llava)) model to generate natural language.
4. **Configuration**: Use of ([Hydra](https://hydra.cc/docs/intro/)) for flexible configuration management.

**NOTE**: The inference is currently slow and for detection, one could use pre-trained YOLO/DETR/ models and then finetune them on specific data from ([OpenImageDataset](https://storage.googleapis.com/openimages/web/index.html)).

## System Design

### Architecture
The system mainly comprises of 5 components

1. **Amenity Schema**: Component responsible to handle the schema the model should follow.
2. **AmenityDetector**: Handles amenity detection and description generation using LLaVA.
3. **AmenityDataManager**: Manages storage, retrieval and summarization of the amenities stored.
4. **PropertyAmenitySystem**: This system processes image/directories, detects amenities and then store the information.
5. **Web App**: Use of streamlit to create and deploy an app.

**NOTE**: There is also an API based component that is responsible in wrapping the model and making it accessible via FastAPI. But this component needs some more work to be done.

### Technical Choices

#### VLM
1. Selected LlaVA as the core VLM due to its strong performance on vision-language tasks
2. Model provides both amenity detection and description generation capabilities
3. ([Git](https://huggingface.co/docs/transformers/en/model_doc/git)), ([InstructBlip](https://huggingface.co/docs/transformers/model_doc/instructblip)) and ([Blip2](https://huggingface.co/docs/transformers/en/model_doc/blip-2)) were also tested out. But either the quality of generation or their inference time served as a disadvantage when comparing with Llava.

#### Data Storage
**SQLite**: Provides a structured, queryable database for complex analyses
**CSV**: Offers easy export and compatibility with other tools

## Getting Started

To get started with the project, follow these steps:

### Cloning the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/Shrinidhibhat87/amenity_detector.git
cd amenity_detector
```

### Setting Up a Python Virtual Environment

Create a Python virtual environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installing Required Libraries

Install the required libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the project

There are two ways to run this project:

### 1. Command Line Approach

1. First, modify the path to where your images are located in the `config/config.yaml` file. Also change the output location to where the data can be stored.
2. Run the main script:
    ```
    python main.py
    ```

### 2. Web-based Approach using Streamlit

1. Launch the Streamlit application:
    ```
    streamlit run streamlit/app.py
    ```
2. Through the web interface, you can upload an image and the application will automatically generate a description for it.

## Limitations and Future Enhancements

### Current Limitations

1. **Slow Inference**: Processing takes approximately 20 seconds per image even with GPU acceleration
2. **Incomplete Detection**: The system cannot reliably detect all amenities present in images
3. **Accuracy Issues**: Some detected amenities are questionable or incorrectly identified
4. **Implicit Amenities**: Cannot detect amenities that aren't visually present (e.g., WiFi, heating systems)

### Planned Enhancements

1. **Optimized Detection Models**: Implement fine-tuned, lightweight models like YOLO/DETR for faster inference, then use language models to describe the detected objects
2. **Retrieval-Augmented Generation**: Incorporate RAG systems that can reference property documentation to enhance descriptions with non-visual amenities
3. **API Infrastructure**: Complete FastAPI integration to provide production-ready endpoints for system access
4. **Agentic AI Implementation**: Develop conversational workflows that allow the model to query and analyze the SQLite database for more comprehensive property insights

## TODO

- [ ] Improve documentation by adding screenshot image of the app working
- [ ] Add pyproject.toml file and .pre-commit.yaml file for better project management.
- [ ] Write up pseudo code as to where the RAG based pipeline would integrate
- [ ] Optimize model inference time by implementing batching
- [ ] Implement FastAPI endpoints for production use
- [ ] Create a Docker container for easy deployment