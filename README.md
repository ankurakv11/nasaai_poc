# Sizing API - Body Measurement and Size Recommendation System

A **standalone** FastAPI-based REST API that provides body measurement extraction from images and garment size recommendations based on user measurements.

**Python 3.8+ Compatible - Completely Self-Contained**

## Requirements

- **Python 3.8 or higher** (tested with Python 3.8, 3.9, 3.10, 3.11)
- **Windows/Linux/macOS** support
- **Minimum 4GB RAM** recommended for image processing
- **Internet connection** for package installation

## Features

- **Standalone Application**: Complete independence - no external dependencies on other projects
- **Image Processing**: Background removal using U2Net neural networks and image enhancement
- **Body Measurement**: MediaPipe-based body measurement extraction from front and side view images
- **Body Measurement Only Mode**: New endpoints that focus only on body measurements without garment fitting
- **Size Recommendations**: Intelligent garment sizing based on measurement data and built-in size charts
- **Multi-format Support**: Handles various image formats and measurement units
- **RESTful API**: Complete REST API with automatic documentation
- **Python 3.8 Compatible**: Uses compatible package versions with comprehensive dependency management
- **Self-Contained Models**: Includes all required neural network models and configuration files

## Project Structure

```
sizing_api/
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Complete project dependencies
├── README.md                   # This file
├── uploads/                    # Temporary file storage
└── app/
    ├── data/                   # Size chart and configuration data
    │   └── Aaiena X UCB.xlsx   # Garment size chart database
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py          # Pydantic models for request/response
    ├── routes/
    │   ├── __init__.py
    │   ├── measurement.py      # Image processing and measurement endpoints
    │   ├── recommendation.py   # Size recommendation endpoints
    │   └── body_measurement.py # Body measurement only endpoints
    ├── services/
    │   ├── __init__.py
    │   ├── image_processing.py # Image enhancement and background removal
    │   ├── measurement.py      # Body measurement extraction
    │   └── recommendation.py   # Size chart analysis and recommendations
    └── utils/                  # Self-contained utilities and models
        ├── bg_removal.py       # Background removal processing
        ├── mediapipe_measurement.py # Complete MediaPipe measurement system
        ├── simple_processing.py # Image processing pipeline
        ├── size_logic.py       # Size analysis and fit calculations
        ├── female.json         # Female body type configurations
        ├── male.json          # Male body type configurations
        ├── libs/              # Neural network libraries
        │   ├── __init__.py
        │   ├── networks.py    # U2Net model implementation
        │   ├── strings.py     # Model configuration
        │   ├── preprocessing.py # Image preprocessing
        │   └── postprocessing.py # Image postprocessing
        └── models/            # Pre-trained models
            └── u2net/
                └── u2net.pth  # U2Net background removal model
```

## Installation

### Prerequisites

1. **Install Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
   ```bash
   python --version  # Should show 3.8.x or higher
   ```

2. **Navigate to the project directory**:
   ```bash
   cd "c:\Users\ehtes\work\software projects\Sizing Code\sizing_api"
   ```

### Quick Start (Windows)

1. **Run the startup script**:
   ```bash
   start_server.bat
   ```
   This will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Start the FastAPI server

### Manual Installation

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/macOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ready to run!**:
   The API is completely self-contained with all required modules, models, and configurations included. No external dependencies needed!
   
   **Note**: The API includes a complete size chart database (`app/data/Aaiena X UCB.xlsx`) and will automatically create a fallback size chart if the Excel file is missing, ensuring the API always works standalone.

## Running the API

Start the FastAPI development server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### Body Measurements Only

#### 1. Get Body Measurements
**POST** `/body-measurements`

Upload and process front and side view images for body measurement extraction only, without garment fitting.

**Request Body** (multipart/form-data):
```
front_image: file (image file)
side_image: file (image file)
height: float (height in cm)
weight: float (weight in kg)
gender: string (M/F)
body_type: string (optional - for females: Triangle, Diamond, Inverted, Rectangle, Hourglass)
```

**Response**:
```json
{
  "success": true,
  "message": "Images processed and body measurements extracted successfully",
  "measurements": {
    "height": 170.5,
    "chest": 92.3,
    "waist": 78.1,
    "hip": 95.2,
    "shoulder_width": 42.5,
    "body_length": 58.7,
    "inseam": 78.0
  },
  "processed_images": {
    "front": "uploads/processed/front_processed.png",
    "side": "uploads/processed/side_processed.png"
  },
  "annotated_images": {
    "front": "uploads/processed/front_annotated.jpg",
    "side": "uploads/processed/side_annotated.jpg"
  }
}
```

#### 2. Get Body Measurements from URLs
**POST** `/body-measurements-from-urls`

Process images from URLs for body measurement extraction only.

**Request Body**:
```json
{
  "front_image_url": "https://example.com/front.jpg",
  "side_image_url": "https://example.com/side.jpg",
  "gender": "F",
  "height": 165.0,
  "weight": 60.0,
  "body_type": "Hourglass"
}
```

**Response**: Same as above endpoint

### Image Processing and Size Recommendations

#### 1. Process Images
**POST** `/process-images`

Upload and process front and side view images for measurement extraction with size recommendations.

**Request Body** (multipart/form-data):
```
front_image: file (image file)
side_image: file (image file)
height: float (height in cm)
weight: float (weight in kg)
gender: string (M/F)
preferred_size: string (optional - S, M, L, XL)
body_type: string (optional - for females: Triangle, Diamond, Inverted, Rectangle, Hourglass)
```

**Response**:
```json
{
  "message": "Images processed successfully",
  "front_image_path": "path/to/processed/front.jpg",
  "side_image_path": "path/to/processed/side.jpg",
  "measurements": {
    "height": 170.5,
    "chest": 92.3,
    "waist": 78.1,
    "hip": 95.2,
    "shoulder": 42.5,
    "sleeve": 58.7
  }
}
```

#### 2. Get User Measurements
**GET** `/measurements`

Retrieve stored user measurements.

#### 3. Save User Measurements
**POST** `/measurements`

Save user measurements manually.

**Request Body**:
```json
{
  "height": 170.5,
  "chest": 92.3,
  "waist": 78.1,
  "hip": 95.2,
  "shoulder": 42.5,
  "sleeve": 58.7,
  "unit": "cm"
}
```

### Size Recommendations

#### 1. Get Categories
**GET** `/categories`

Get available garment categories.

#### 2. Get Products by Category
**GET** `/categories/{category}/products`

Get products available in a specific category.

#### 3. Get Sizes for Product
**GET** `/products/{product}/sizes`

Get available sizes for a specific product.

#### 4. Analyze Fit
**POST** `/analyze-fit`

Analyze how well a specific size fits the user's measurements.

**Request Body**:
```json
{
  "product_name": "shirt",
  "size": "M",
  "measurements": {
    "height": 170.5,
    "chest": 92.3,
    "waist": 78.1,
    "hip": 95.2,
    "shoulder": 42.5,
    "sleeve": 58.7,
    "unit": "cm"
  }
}
```

**Response**:
```json
{
  "product_name": "shirt",
  "size": "M",
  "fit_analysis": {
    "overall_rating": "Good Fit",
    "fit_percentage": 85.5,
    "detailed_analysis": {
      "chest": "Perfect fit",
      "waist": "Slightly loose",
      "shoulder": "Good fit"
    }
  }
}
```

#### 5. Get Best Fit
**POST** `/best-fit/{product}`

Get the best fitting size for a product based on stored measurements.

## Data Models

### UserMeasurements
```python
{
  "height": float,      # Height in specified unit
  "chest": float,       # Chest circumference
  "waist": float,       # Waist circumference
  "hip": float,         # Hip circumference
  "shoulder": float,    # Shoulder width
  "sleeve": float,      # Sleeve length
  "unit": str          # "cm" or "inch"
}
```

### FitAnalysisResponse
```python
{
  "product_name": str,
  "size": str,
  "fit_analysis": {
    "overall_rating": str,        # "Excellent", "Good Fit", "Acceptable", "Poor Fit"
    "fit_percentage": float,      # 0-100
    "detailed_analysis": dict     # Per-measurement analysis
  }
}
```

## Configuration

The API uses the following default configurations:

- **Upload Directory**: `./uploads/` (automatically created)
- **Supported Image Formats**: JPEG, PNG, JPG
- **Maximum File Size**: Configured by FastAPI defaults
- **CORS**: Enabled for all origins (development mode)

## Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid input data or missing required fields
- **404 Not Found**: Resource not found (product, size chart, etc.)
- **422 Unprocessable Entity**: Invalid request body format
- **500 Internal Server Error**: Processing errors with detailed messages

## Development

### Adding New Endpoints

1. Define Pydantic models in `app/models/schemas.py`
2. Implement business logic in appropriate service files
3. Create API routes in `app/routes/`
4. Register routes in `main.py`

### Testing

Run the development server and use the interactive documentation at `/docs` to test endpoints, or use tools like:

- **curl**: Command line HTTP client
- **Postman**: GUI-based API testing
- **httpx**: Python HTTP client for automated testing

## Dependencies

Key dependencies include:

- **FastAPI**: Modern, fast web framework
- **MediaPipe**: Body pose detection and measurement
- **OpenCV**: Image processing and computer vision
- **PyTorch**: Deep learning framework for U2Net background removal
- **scikit-image**: Advanced image processing algorithms
- **Pandas**: Data manipulation for size charts
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for FastAPI
- **NumPy**: Numerical computing
- **Pillow (PIL)**: Image handling and manipulation

All dependencies are specified in `requirements.txt` with Python 3.8+ compatible versions.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Ensure at least 4GB RAM available for neural network processing
2. **Image Processing Errors**: Check image format and file size - supported formats: JPEG, PNG, JPG
3. **Measurement Extraction Failures**: Ensure clear, well-lit images with full body visible in standing pose
4. **Model Loading Errors**: Verify that `app/utils/models/u2net/u2net.pth` exists and is not corrupted
5. **Import Errors**: All required modules are self-contained - if issues persist, reinstall dependencies
6. **Size Chart Missing**: The API includes fallback size chart data if the Excel file is missing - no external files required

### Performance Optimization

- **GPU Support**: PyTorch will automatically use CUDA if available for faster background removal
- **Memory Management**: Large images are automatically resized for optimal processing
- **Concurrent Processing**: API supports multiple simultaneous requests

### Logs

Check the console output for detailed error messages and processing information.

## License

This is a standalone body measurement and size recommendation API system. The project includes self-contained implementations of image processing, body measurement, and size analysis algorithms.

---

## Technical Architecture

### Standalone Design
This API has been designed to be completely self-contained:
- **No External Dependencies**: All required modules are included in the `app/utils/` directory
- **Portable**: The entire `sizing_api` folder can be moved or deployed independently
- **Self-Sufficient**: Includes pre-trained models, configuration files, and all processing logic

### Processing Pipeline
1. **Image Upload**: Accepts front and side view images
2. **Background Removal**: Uses U2Net neural network for precise background removal
3. **Pose Detection**: MediaPipe extracts body landmarks and key points
4. **Measurement Calculation**: Anthropometric calculations based on pose landmarks
5. **Size Analysis**: Compares measurements against built-in size charts for recommendations

### Data Security
- **Local Processing**: All image processing happens locally - no data sent to external services
- **Temporary Storage**: Uploaded images are stored temporarily and can be automatically cleaned up
- **Privacy First**: No personal data is permanently stored without explicit user consent