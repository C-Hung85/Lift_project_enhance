# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a lift motion detection system that analyzes elevator movement from video files using computer vision and feature detection. The system processes video files to detect vertical movement, applies scale calibration, and generates motion analysis results with manual correction capabilities.

## Key Development Commands

### Environment Setup
- **Python Version**: 3.10.13 (required)
- **Package Manager**: Use `uv` (preferred) or `pipenv` for dependency management
- **Virtual Environment**: Always activate environment before development

```bash
# Using uv (recommended)
uv sync
uv run python <script>

# Using pipenv (alternative)
pipenv install
pipenv shell
python <script>
```

### Core Analysis Pipeline
- **Main Analysis**: `uv run python src/lift_travel_detection.py`
- **Manual Correction Tool**: `uv run python src/manual_correction_tool.py`
- **Data Cleaning**: `uv run python data_cleaner.py`

### Setup and Configuration Tools
- **Rotation Setup**: `uv run python src/rotation_setup.py` - Configure video rotation correction
- **Scale Setup**: `uv run python src/scale_setup.py` - Manage scale calibration cache
- **Scale Cache Building**: `uv run python src/build_scale_cache.py` - Build scale calibration

### Development and Debug Tools
- **Frame Export**: `uv run python src/export_frames.py` - Extract specific frames
- **Debug Analysis**: `uv run python debug_cluster_issue.py`
- **Single Value Analysis**: `uv run python analyze_single_values.py`

## Project Architecture

### Core Processing Pipeline
```
Video Files → Rotation Correction → Feature Detection → Motion Analysis → Scale Conversion → Results Output
    ↓              ↓                    ↓               ↓               ↓              ↓
  data/         rotation_            ORB features    K-means         scale_        CSV + Video
  *.mp4         config.py           + matching       clustering      config.py     inspection
```

### Directory Structure
```
lifts/
├── data/           # Input video files (*.mp4)
├── scale_images/   # Scale reference images (with red markers)
├── inspection/     # Output inspection videos
└── result/         # CSV analysis results
    ├── *.csv           # Raw analysis results
    ├── c*.csv          # Cleaned data (noise removed)
    └── mc*.csv         # Manually corrected data
```

### Configuration System
- **`src/config.py`**: Main configuration, video timing parameters, ROI settings
- **`src/rotation_config.py`**: Video rotation angles (auto-generated)
- **`src/scale_config.py`**: Pixel-to-mm scale factors (auto-generated cache)
- **`src/darkroom_intervals.py`**: Time intervals to ignore (optional)

### Key Processing Modules

#### Motion Detection (`src/lift_travel_detection.py`)
- Uses ORB feature detection with BF matcher
- K-means clustering for motion separation
- Statistical significance testing (t-test)
- Frame-by-frame analysis with configurable intervals
- Scale calibration integration

#### Manual Correction System (`src/manual_correction_tool.py`)
- Semi-automatic GUI-based correction tool
- ROI selection and 8x zoom precision marking
- Reference line segment marking (anti-shake)
- Proportional displacement correction
- Handles equipment failure detection

#### Rotation Correction System
- **`src/rotation_utils.py`**: Core rotation functions
- **`src/rotation_setup.py`**: Interactive setup tool
- Rigid rotation with black fill
- Synchronized video and scale image rotation

#### Scale Calibration System
- **`src/scale_cache_utils.py`**: Cache management utilities
- **`src/build_scale_cache.py`**: Build calibration cache
- Red marker detection in scale images
- Automatic cache invalidation and updates

## Data Flow and Processing

### 1. Video Preprocessing
- Load video files from `lifts/data/`
- Apply rotation correction if configured
- Extract frames at specified intervals (default: every 6 frames)

### 2. Feature Detection and Matching
- ORB feature detection within ROI mask
- BF matcher for feature correspondence
- Outlier removal using statistical methods
- Camera pan detection using t-test

### 3. Motion Analysis
- K-means clustering (2 clusters) for motion separation
- Vertical displacement calculation
- Statistical significance testing (p < 0.0001)
- Darkroom interval filtering (optional)

### 4. Scale Conversion
- Pixel-to-millimeter conversion using scale factors
- Formula: `displacement_mm = (pixel_diff * 10.0) / scale_factor`
- Automatic scale cache management

### 5. Output Generation
- CSV files with frame indices, timestamps, and displacement values
- Inspection videos with motion visualization
- Cleaning reports in JSON format

## Development Patterns

### Configuration Management
- Use environment variables for paths (`SATA` for data folder)
- Video-specific settings in `video_config` dictionary
- Graceful handling of missing configuration files

### Error Handling and Logging
- Probe mode for debugging (`LIFT_PROBE` environment variable)
- Comprehensive error messages with video context
- Automatic fallback for missing dependencies

### Cache and Performance
- Scale calibration caching for performance
- Directory hash-based cache invalidation
- Lazy loading of optional configurations

### GUI Development (Tkinter)
- Event-driven architecture for user interactions
- Canvas-based image display with coordinate transformations
- Keyboard shortcuts for workflow efficiency

## Testing and Validation

### Manual Testing
- Use specific video files with `LIFT_TARGET` environment variable
- Frame-specific debugging with `LIFT_PROBE_FRAMES`
- Visual inspection through generated inspection videos

### Data Validation
- Statistical significance tests for motion detection
- Outlier removal for robust analysis
- Noise filtering in data cleaning pipeline

## Common Development Tasks

### Adding New Video Configuration
1. Add entry to `video_config` in `src/config.py`
2. Run rotation setup if needed: `uv run python src/rotation_setup.py`
3. Ensure scale images exist in `lifts/scale_images/`

### Debugging Motion Detection Issues
1. Enable probe mode: `set LIFT_PROBE=1`
2. Set specific frames: `set LIFT_PROBE_FRAMES=1000,2000,3000`
3. Run with target video: `set LIFT_TARGET=video.mp4`

### Manual Correction Workflow
1. Run data cleaning: `uv run python data_cleaner.py`
2. Start correction tool: `uv run python src/manual_correction_tool.py`
3. Select cleaned CSV file (c*.csv)
4. Follow GUI workflow: ROI selection → line marking → correction

### Performance Optimization
- Use scale cache system for faster startup
- Configure appropriate frame intervals in `config.py`
- Optimize ROI settings to reduce processing area

## Important Dependencies
- **OpenCV**: Core computer vision functionality
- **NumPy**: Numerical operations and array handling
- **Pandas**: Data manipulation and CSV handling
- **Scikit-learn**: K-means clustering and statistical tools
- **EasyOCR**: Optional OCR functionality
- **Matplotlib/Seaborn**: Data visualization
- **Streamlit**: Web interface components
- **Tkinter**: GUI framework for manual correction

## Environment Variables
- `SATA`: Data folder path (defaults to current directory)
- `LIFT_PROBE`: Enable debug logging (set to '1')
- `LIFT_PROBE_INTERVAL`: Frame interval for probe logging
- `LIFT_PROBE_FRAMES`: Specific frames for detailed logging
- `LIFT_TARGET`: Process only specific video file