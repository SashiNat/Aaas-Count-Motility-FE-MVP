# Sperm Analyzer - WHO 2021 Motility Analysis Tool

### **📋 Overview**

The **Sperm Analyzer** is a post-processing tool that analyzes sperm motility from CenterTrack detection results. After successful training and testing of the CenterTrack model on sperm videos, this script processes the tracking data to provide WHO 2021 compliant motility assessments.

### **🔄 Workflow Pipeline**

```
CenterTrack Training → CenterTrack Testing → Sperm Analyzer → WHO Report
     (Model)              (JSON + Video)        (Analysis)      (Results)
```

#### **Prerequisites: CenterTrack Results**

Before using the Sperm Analyzer, you must have completed:

1. **CenterTrack Training**: Trained model on sperm dataset
2. **CenterTrack Testing**: Generated detection results from test videos

**Required CenterTrack Outputs:**
- `{sample_id}_results.json` - Tracking data with bounding boxes and track IDs
- `{sample_id}.mp4` - Original video with detections (optional, for enhanced output)

### **🧬 How Sperm Analyzer Works**
---

#### **Input Processing**
1. **Reads JSON Results**: Loads CenterTrack tracking data
2. **Extracts Tracks**: Organizes detections by tracking ID across frames
3. **Auto-detects Properties**: Determines video FPS and resolution from files

#### **Speed Calculation**
```python
# For each sperm track:
distance_pixels = sqrt((x2-x1)² + (y2-y1)²)  # Between consecutive frames
time_seconds = frame_difference / fps          # Time elapsed
speed_pixels_per_sec = distance_pixels / time_seconds
speed_micrometers_per_sec = speed_pixels_per_sec / pixels_per_micrometer
```

#### **WHO 2021 Classification**
Based on calculated speeds, each sperm is classified into:

| Category | Speed Range | Color Code | Description |
|----------|-------------|------------|-------------|
| **RP** (Rapid Progressive) | ≥25 μm/s | 🟢 Green | Fast, forward movement |
| **SP** (Slow Progressive) | 5-24.9 μm/s | 🟡 Yellow | Moderate, forward movement |
| **NP** (Non-Progressive) | 0.5-4.9 μm/s | 🔵 Blue | Movement without progression |
| **IM** (Immotile) | <0.5 μm/s | ⚫ Gray | No significant movement |

### **🚀 Usage**
---

#### **Basic Analysis**
```bash
!python sperm_analyzer.py \
  --json_file ./CenterTrack_Fixed/results/mot17_half_sc_ME_001_results.json \
  --video_file ./CenterTrack_Fixed/videos/ME_001.wmv \
  --pixels_per_um 1.0 \
  --output_dir ./CenterTrack_Fixed/results/analysis_report
```

#### **Key Parameters**
- `--json_file`: CenterTrack results (required)
- `--video_file`: Original video for enhanced output (optional)
- `--pixels_per_um`: Conversion ratio based on microscope setup (required)
- `--output_dir`: Where to save analysis results (optional)

### **📊 Expected Results**
---

#### **1. WHO Motility Report** - `{sample_id}_WHO_motility_report.txt`
```
WHO 2021 Sperm Motility Analysis Report
======================================
Analysis of: CR_001_results.json
Sample ID: CR_001
Generated: 2024-12-19 14:30:25

Total Sperm Analyzed: 156

Individual Categories:
  1. Rapid Progressive (RP): 18 (11.5%)
  2. Slow Progressive (SP): 87 (55.8%)
  3. Non-Progressive (NP): 23 (14.7%)
  4. Immotile (IM): 28 (17.9%)

WHO Standard Categories:
  1. Total Motile (RP+SP+NP): 128 (82.1%)
  2. Total Progressive (RP+SP): 105 (67.3%)
  3. Rapid Progressive (RP): 18 (11.5%)
  4. Slow Progressive (SP): 87 (55.8%)
  5. Non-Progressive (NP): 23 (14.7%)
  6. Immotile (IM): 28 (17.9%)

WHO Reference Values (Normal):
  - Total Motile: ≥40%
  - Progressive Motile: ≥32%
  - Rapid Progressive: No specific threshold
```

#### **2. Detailed Analysis Data** - `{sample_id}_detailed_analysis.json`
```json
{
  "analysis_info": {
    "input_json_file": "./results/CR_001_results.json",
    "input_video_file": "./videos/CR_001.wmv",
    "output_prefix": "CR_001",
    "fps": 30.0,
    "resolution": {"width": 640, "height": 384},
    "pixels_per_micrometer": 5.33,
    "total_tracks": 156,
    "valid_tracks": 128
  },
  "tracks": {
    "1": {
      "classification": "SP",
      "speed_data": {
        "avg_speed": 12.4,
        "max_speed": 18.7,
        "track_length": 45,
        "duration": 1.5
      },
      "positions": [...]
    }
  }
}
```

#### **3. Color-Coded Video** - `{sample_id}_classified_sperm_video.mp4`
- **Visual Output**: Original video with color-coded bounding boxes
- **Real-time Classification**: Each sperm shown with its motility category
- **Track IDs**: Numerical identifiers for each sperm
- **Color Legend**: 
  - 🟢 Green (RP) = Fast swimmers
  - 🟡 Yellow (SP) = Moderate swimmers  
  - 🔵 Blue (NP) = Non-progressive movement
  - ⚫ Gray (IM) = Immotile

### **📄 Output Summary**

After running the Sperm Analyzer, you will have:

✅ **Clinical Report**: WHO 2021 compliant motility assessment  
✅ **Detailed Data**: Complete speed and tracking information  
✅ **Visual Verification**: Color-coded video for result validation  
✅ **Quality Metrics**: Statistical confidence and analysis parameters  

The Sperm Analyzer transforms raw CenterTrack detections into clinically meaningful motility assessments suitable for medical diagnosis and research applications.
