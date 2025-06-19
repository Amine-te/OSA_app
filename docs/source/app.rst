=============================================
Intelligent Retail Shelf Analysis App
=============================================

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

The Intelligent Retail Shelf Analysis is a comprehensive computer vision application built with Streamlit that analyzes retail shelf images and videos to provide inventory insights, stock level monitoring, and restocking recommendations. The application leverages multiple AI models including YOLO for object detection, CNN for product classification, and specialized void detection models.

Features
========

Core Functionality
-----------------

* **Image Analysis**: Single image analysis with complete product detection and inventory assessment
* **Video Analysis**: Frame-by-frame video processing with temporal trend analysis
* **Multi-Model Pipeline**: Integration of YOLO, CNN, and void detection models
* **Real-time Visualization**: Interactive charts, gauges, and detection overlays
* **Export Capabilities**: JSON, CSV, and detailed text report generation

Analysis Capabilities
--------------------

* **Product Detection**: Automatic identification of products on shelves
* **Stock Level Assessment**: Calculation of current stock percentages
* **Void Detection**: Identification of empty shelf spaces
* **Inventory Tracking**: Missing product estimation and capacity analysis
* **Trend Analysis**: Temporal stock level changes in video sequences

Visualization Features
---------------------

* **Detection overlays**: Bounding boxes with confidence scores
* **Stock level gauges**: Interactive circular progress indicators
* **Trend charts**: Time-series analysis of stock levels
* **Comparative analysis**: Frame-by-frame comparison tools
* **Export visualizations**: Downloadable analysis charts

Installation
============

Prerequisites
------------

Before installing the application, ensure you have the following dependencies:

.. code-block:: bash

   pip install streamlit
   pip install opencv-python
   pip install numpy
   pip install pillow
   pip install plotly
   pip install pandas
   pip install matplotlib

Required Files
--------------

The application requires the following model files and directory structure:

.. code-block::

   project_root/
   ├── app.py                              # Main Streamlit application
   ├── pipeline.py                         # EnhancedRetailPipeline class
   ├── models/
   │   ├── sku/
   │   │   └── individual_products.pt      # YOLO model for product detection
   │   ├── classifier/
   │   │   ├── best_lightweight_cnn.pth    # CNN model for classification
   │   │   └── model_info.json             # Class names and model metadata
   │   └── void/
   │       └── void_0,95_best_one.pt       # Void detection model
   └── requirements.txt

Usage
=====

Starting the Application
-----------------------

1. **Launch the Streamlit app**:

   .. code-block:: bash

      streamlit run app.py

2. **Access the web interface** at ``http://localhost:8501``

Configuration
-------------

Pipeline Initialization
^^^^^^^^^^^^^^^^^^^^^^

Before analyzing any media, configure the pipeline in the sidebar:

1. **Model Paths**:
   
   * **YOLO Model Path**: Path to your YOLO model file (default: ``models/sku/individual_products.pt``)
   * **CNN Model Path**: Path to your CNN classification model (default: ``models/classifier/best_lightweight_cnn.pth``)
   * **Void Model Path**: Path to your void detection model (default: ``models/void/void_0,95_best_one.pt``)

2. **Product Classes**:
   
   * The application automatically loads class names from ``models/classifier/model_info.json``
   * Alternatively, manually enter comma-separated class names (e.g., ``cocacola,oil,water``)

3. **Detection Thresholds**:
   
   * **Product Detection Confidence**: Minimum confidence for product detection (default: 0.5)
   * **Void Detection Confidence**: Minimum confidence for void detection (default: 0.3)

4. **Initialize Pipeline**: Click the "🚀 Initialize Pipeline" button to load all models

Image Analysis
--------------

Single Image Processing
^^^^^^^^^^^^^^^^^^^^^^

1. **Upload Image**:
   
   * Navigate to the "📷 Image Analysis" tab
   * Upload an image file (supported formats: JPG, JPEG, PNG, BMP)
   * The original image will be displayed for preview

2. **Run Analysis**:
   
   * Click "🔍 Analyze Image" to process the image
   * The system will detect products, classify them, and identify voids
   * Results include detection visualization, stock metrics, and detailed analysis

3. **Review Results**:
   
   * **Overview Metrics**: Total products, missing products, overall stock percentage
   * **Stock Level Analysis**: Bar charts showing current vs. missing inventory
   * **Stock Level Gauges**: Circular progress indicators for each product
   * **Detailed Product Information**: Expandable sections with complete product data
   * **Summary Table**: Comprehensive tabular view of all products

Video Analysis
--------------

Frame-by-Frame Processing
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Upload Video**:
   
   * Navigate to the "🎥 Video Analysis" tab
   * Upload a video file (supported formats: MP4, AVI, MOV, MKV)
   * The video player will display for preview

2. **Configure Processing**:
   
   * **Frame Interval**: Extract frames every N seconds (default: 5)
   * **Max Frames**: Maximum number of frames to analyze (default: 10)
   * **Analysis Mode**: Choose between "Frame-by-Frame" or "Trend Analysis"

3. **Process Video**:
   
   * Click "🎬 Analyze Video" to begin processing
   * Progress bar shows extraction and analysis progress
   * Each frame is processed through the complete pipeline

4. **Navigate Results**:
   
   * **Frame Navigation**: Use First/Previous/Next/Last buttons
   * **Frame Selector**: Dropdown to jump to specific frames
   * **Individual Frame Analysis**: Complete analysis for each frame
   * **Frame Comparison**: Quick overview table of all frames

Trend Analysis
^^^^^^^^^^^^^^

1. **Overall Trends**:
   
   * **Stock Level Trend**: Time-series chart of overall stock percentage
   * **Products Detected**: Trend of total products detected over time
   * **Threshold Lines**: Visual indicators for good (90%) and warning (70%) levels

2. **Product-Specific Trends**:
   
   * Individual trend charts for each product type
   * Stock level changes over time for specific products
   * Comparative analysis across different product categories

Export Options
--------------

The application provides multiple export formats:

JSON Export
^^^^^^^^^^^

* Complete analysis results in JSON format
* Includes all detection data, classifications, and metadata
* Suitable for programmatic processing and integration

CSV Export
^^^^^^^^^^^

* Tabular data with product information
* Columns: Product, Current_Count, Missing_Count, Full_Capacity, Stock_Percentage
* Compatible with spreadsheet applications

Detailed Report
^^^^^^^^^^^^^^

* Human-readable text report
* Includes timestamp, overview metrics, and product inventory
* Formatted for easy reading and sharing

Visualization Export
^^^^^^^^^^^^^^^^^^^

* Downloadable PNG images of detection visualizations
* Includes bounding boxes, confidence scores, and analysis charts
* High-resolution images suitable for presentations

API Reference
=============

Core Components
---------------

EnhancedRetailPipeline
^^^^^^^^^^^^^^^^^^^^^

The main processing pipeline class that orchestrates all analysis functions.

**Initialization Parameters**:

* ``yolo_model_path`` (str): Path to YOLO model file
* ``cnn_model_path`` (str): Path to CNN classification model
* ``void_model_path`` (str): Path to void detection model
* ``class_names`` (list): List of product class names
* ``confidence_threshold`` (float): Minimum confidence for product detection
* ``void_confidence_threshold`` (float): Minimum confidence for void detection

**Key Methods**:

* ``detect_and_classify_complete(image_path)`` → dict: Complete analysis of single image
* ``visualize_complete_results(results, save_path)`` → None: Generate visualization

Session State Variables
^^^^^^^^^^^^^^^^^^^^^^

The application maintains several session state variables:

* ``st.session_state.pipeline``: Initialized pipeline instance
* ``st.session_state.results``: Current analysis results
* ``st.session_state.current_image_path``: Path to current image being analyzed
* ``st.session_state.video_frames_data``: List of video frame analysis results
* ``st.session_state.current_frame_index``: Current frame index for navigation
* ``st.session_state.temp_dir``: Temporary directory for file operations

Configuration Options
=====================

Model Configuration
------------------

The application supports various model configurations:

YOLO Model Settings
^^^^^^^^^^^^^^^^^^

* **Model Format**: PyTorch (.pt) format
* **Input Size**: Configurable based on model training
* **Confidence Threshold**: Adjustable detection confidence (0.1-1.0)
* **NMS Threshold**: Non-maximum suppression threshold

CNN Classification Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Model Format**: PyTorch (.pth) format
* **Input Preprocessing**: Automatic image normalization and resizing
* **Class Names**: Loaded from model_info.json or manual configuration
* **Batch Processing**: Optimized for single and batch inference

Void Detection Settings
^^^^^^^^^^^^^^^^^^^^^^

* **Model Format**: PyTorch (.pt) format
* **Confidence Threshold**: Adjustable void detection sensitivity
* **Integration**: Seamless integration with product detection pipeline

Performance Optimization
========================

Memory Management
----------------

The application implements several memory optimization strategies:

* **Temporary File Cleanup**: Automatic removal of temporary files
* **Session-based Storage**: Efficient session state management
* **Progressive Loading**: Lazy loading of large models and data

Processing Optimization
----------------------

* **Batch Processing**: Efficient handling of multiple frames
* **Progress Tracking**: Real-time progress indicators for long operations
* **Error Handling**: Robust error handling and recovery mechanisms

Storage Considerations
---------------------

* **Temporary Files**: Automatic cleanup of temporary image and video files
* **Session Persistence**: Results persist within browser session
* **Export Formats**: Multiple export options to minimize storage requirements

Troubleshooting
===============

Common Issues
------------

Model Loading Errors
^^^^^^^^^^^^^^^^^^^^

**Problem**: Pipeline initialization fails with model loading errors.

**Solutions**:

* Verify model file paths are correct
* Ensure model files exist and are readable
* Check model format compatibility (PyTorch .pt/.pth files)
* Verify sufficient system memory for model loading

Memory Issues
^^^^^^^^^^^^

**Problem**: Application crashes or becomes unresponsive during processing.

**Solutions**:

* Reduce max frames for video processing
* Increase frame interval to process fewer frames
* Use smaller input images
* Restart the application to clear memory

File Upload Issues
^^^^^^^^^^^^^^^^^

**Problem**: Image or video upload fails or produces errors.

**Solutions**:

* Verify file format is supported
* Check file size limitations
* Ensure file is not corrupted
* Try converting to a different supported format

Processing Errors
^^^^^^^^^^^^^^^^

**Problem**: Analysis fails with processing errors.

**Solutions**:

* Check model paths and file permissions
* Verify class names match model training
* Adjust confidence thresholds
* Review error messages for specific issues

Performance Issues
^^^^^^^^^^^^^^^^^

**Problem**: Slow processing or analysis times.

**Solutions**:

* Reduce image resolution for faster processing
* Decrease confidence thresholds if appropriate
* Process fewer video frames
* Use GPU acceleration if available

Best Practices
==============

Model Management
---------------

* **Version Control**: Keep track of model versions and performance metrics
* **Backup Models**: Maintain backups of trained models
* **Documentation**: Document model training parameters and performance
* **Testing**: Regularly test models with new data

Data Handling
------------

* **Input Quality**: Use high-quality, well-lit images for best results
* **Consistent Lighting**: Maintain consistent lighting conditions
* **Camera Angles**: Use consistent camera angles and distances
* **Regular Updates**: Update models with new product types and shelf configurations

Application Deployment
----------------------

* **Resource Allocation**: Ensure sufficient CPU/GPU resources
* **Monitoring**: Implement monitoring for performance and errors
* **Scaling**: Plan for horizontal scaling if needed
* **Security**: Implement appropriate security measures for production use

Data Privacy
-----------

* **Image Handling**: Implement secure image processing and storage
* **Data Retention**: Define data retention policies
* **Access Control**: Implement appropriate access controls
* **Compliance**: Ensure compliance with relevant data protection regulations

Future Enhancements
==================

Planned Features
---------------

* **Real-time Processing**: Live camera feed analysis
* **Database Integration**: Persistent storage of analysis results
* **API Endpoints**: RESTful API for programmatic access
* **Mobile Support**: Mobile-optimized interface
* **Advanced Analytics**: Machine learning-based trend prediction
* **Multi-store Support**: Analysis across multiple store locations
* **Automated Alerts**: Real-time notifications for low stock levels

Technical Improvements
---------------------

* **GPU Acceleration**: Enhanced GPU support for faster processing
* **Distributed Processing**: Multi-node processing capabilities
* **Model Optimization**: Improved model efficiency and accuracy
* **Caching**: Intelligent caching for repeated analyses
* **Streaming**: Real-time video stream processing

License and Support
===================

License
-------

This application is provided as-is for educational and commercial use. Please refer to the specific license terms for your implementation.

Support
-------

For technical support, feature requests, or bug reports:

* Review the troubleshooting section
* Check the GitHub repository for updates
* Contact the development team for enterprise support

Contributing
-----------

Contributions are welcome! Please follow the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
5. Provide comprehensive testing and documentation

Version History
===============

Current Version: 1.0.0
----------------------

* Initial release with core functionality
* Image and video analysis capabilities
* Export options and visualization tools
* Comprehensive documentation and troubleshooting guides

Acknowledgments
===============

This application was built using the following open-source libraries and frameworks:

* **Streamlit**: Web application framework
* **OpenCV**: Computer vision library
* **NumPy**: Numerical computing library
* **Pillow**: Image processing library
* **Plotly**: Interactive visualization library
* **Pandas**: Data manipulation library
* **Matplotlib**: Plotting library

Special thanks to the open-source community for providing the tools and libraries that make this application possible.