# CenterTrack Repository Modifications - Complete Technical Documentation

### 📋 Overview

This document provides a comprehensive technical breakdown of all modifications made to the original CenterTrack repository to achieve:
1. **Training Compatibility**: CUDA 11.7, PyTorch 1.13.1, Azure VM environments
2. **Testing/Inference**: Headless operation and sperm analysis functionality
3. **Modern Environment Support**: Updated dependencies and error handling

#### **Original Repository**
- **Source**: https://github.com/xingyizhou/CenterTrack
- **Paper**: [Tracking Objects as Points (ECCV 2020)](https://arxiv.org/abs/2004.01177)
- **Original Target**: PyTorch 1.0-1.12, older CUDA versions

### **🚨 Complete Modification Overview**


| Category | Files Changed | Files Added | Impact | Phase |
|----------|---------------|-------------|---------|-------|
| **DCNv2 Implementation** | 0 | 2 | 🔴 Critical | Training |
| **Import Compatibility** | 1 | 0 | 🟡 Medium | Training |
| **Headless Operation** | 3 | 0 | 🟢 Enhancement | Testing |
| **Enhanced test output control** | 1 | 0 | 🟢 Enhancement | Testing |
| **Git Integration** | 0 | 1 | 🟢 Enhancement | Both |


### **🔴 CRITICAL: Training Environment Modifications**
---

### **1. Custom DCNv2 Implementation**

#### **Problem Statement**
The original CenterTrack repository uses DCNv2 (Deformable Convolution Networks v2) which has critical compatibility issues:
- **Compilation Failures**: Original DCNv2 fails to compile with PyTorch 1.13+
- **Missing Headers**: References `TH/TH.h` and `THC/THC.h` which were removed in newer PyTorch versions
- **API Changes**: Parameter naming conventions differ between versions
- **Build System Issues**: Makefile and setup.py incompatible with modern environments

#### **Files Added - Training Phase**

##### **File 1**: `src/lib/model/networks/DCNv2/__init__.py`
```python
# DCNv2 package wrapper using mmcv for CenterTrack compatibility
from .dcn_v2 import DCN, DCNv2

# Also make them available at package level
__all__ = ['DCN', 'DCNv2']
```

##### **File 2**: `src/lib/model/networks/DCNv2/dcn_v2.py`
```python
# dcn_v2.py - Compatibility module for CenterTrack
from mmcv.ops import DeformConv2dPack as _DeformConv2dPack
from mmcv.ops import ModulatedDeformConv2dPack as _ModulatedDeformConv2dPack
import torch.nn as nn
import torch

class DCN(nn.Module):
    """Deformable Convolution wrapper compatible with original DCNv2"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, deform_groups=None, 
                 deformable_groups=None, bias=True):
        super(DCN, self).__init__()
        
        # Handle both parameter names: deform_groups (mmcv) and deformable_groups (original DCNv2)
        if deformable_groups is not None:
            deform_groups = deformable_groups
        elif deform_groups is None:
            deform_groups = 1
        
        # Use DeformConv2dPack but without bias (mmcv limitation)
        self.dcn = _DeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False  # Always False for mmcv
        )
        
        # Add bias manually if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # DeformConv2dPack handles offset generation internally
        out = self.dcn(x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

class DCNv2(nn.Module):
    """Modulated Deformable Convolution v2 wrapper compatible with original DCNv2"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, deform_groups=None,
                 deformable_groups=None, bias=True):
        super(DCNv2, self).__init__()
        
        # Handle both parameter names
        if deformable_groups is not None:
            deform_groups = deformable_groups
        elif deform_groups is None:
            deform_groups = 1
        
        # Use ModulatedDeformConv2dPack but without bias (mmcv limitation)
        self.dcn = _ModulatedDeformConv2dPack(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups,
            bias=False  # Always False for mmcv
        )
        
        # Add bias manually if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # ModulatedDeformConv2dPack handles offset and mask generation internally
        out = self.dcn(x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

# Make DCN the default export that CenterTrack expects
__all__ = ['DCN', 'DCNv2']
```

#### **Technical Implementation Details**

##### **1.1. Parameter Name Compatibility**
```python
# Original DCNv2 uses 'deformable_groups'
# mmcv uses 'deform_groups'
# Our wrapper handles both:

if deformable_groups is not None:
    deform_groups = deformable_groups
elif deform_groups is None:
    deform_groups = 1
```

##### **1.2. Bias Handling Workaround**
```python
# Problem: mmcv DeformConv2d doesn't support bias=True
# Solution: Manual bias implementation

self.dcn = _DeformConv2dPack(..., bias=False)  # Force False
if bias:
    self.bias = nn.Parameter(torch.zeros(out_channels))  # Manual bias

# In forward():
out = self.dcn(x)
if self.bias is not None:
    out = out + self.bias.view(1, -1, 1, 1)  # Add bias manually
```

##### **1.3. mmcv Dependency**
- **Requirement**: `mmcv-full==1.7.1` with CUDA 11.7 support
- **Installation**: `pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html`
- **Compatibility**: Tested with PyTorch 1.13.1+cu117

### **2. Import Compatibility Fix**
---

#### **File Modified**: `src/lib/model/networks/backbones/mobilenet.py`

```python
# Before (Line 14)
from torchvision.models.utils import load_state_dict_from_url

# After (Line 14)
from torch.hub import load_state_dict_from_url
```

**Reason**: `torchvision.models.utils` was deprecated and removed in torchvision 0.14+



### **🟢 Testing/Inference Phase Modifications**
---

### **3. Headless Operation Support**

#### **File Modified**: `src/demo.py` - Complete Rewrite for Enhanced Functionality

**Key Enhancements**:

##### **3.1. Auto-Detection Functions (NEW)**
```python
def get_video_info(video_path):
    """Get video properties like fps, width, height"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def process_single_video(opt, video_path, output_dir, detector):
    """Process a single video file with auto property detection"""
    fps, width, height = get_video_info(video_path)
    opt.save_framerate = fps
    opt.video_w = width  
    opt.video_h = height
    # ... processing logic
```

##### **3.2. Batch Directory Processing (NEW)**
```python
# Enhanced directory processing with multiple formats
if os.path.isdir(opt.demo):
    video_files = []
    for ext in video_ext:
        pattern = os.path.join(opt.demo, f"*.{ext}")
        video_files.extend(glob.glob(pattern))
        # Also check uppercase extensions
        pattern = os.path.join(opt.demo, f"*.{ext.upper()}")
        video_files.extend(glob.glob(pattern))
    
    for video_file in sorted(video_files):
        process_single_video(opt, video_file, output_dir, detector)
```

##### **3.3. Extended Video Format Support**
```python
# Original: Limited video formats
video_ext = ['mp4', 'mov', 'avi', 'mkv']

# Modified: Extended format support
video_ext = ['mp4', 'mov', 'avi', 'mkv', 'wmv']  # Added wmv
```

##### **3.4. Headless Operation Handling**
```python
# Remove all GUI dependencies for headless servers
# REMOVED: cv2.imshow(), cv2.waitKey()
# ADDED: Progress logging and error handling

if opt.debug > 0:
    try:
        cv2.imshow('input', img)
        if cv2.waitKey(1) == 27:  # ESC key
            save_and_exit(opt, out, results, 'webcam')
            return
    except:
        print("Display not available, running in headless mode")
        opt.debug = 0  # Disable further display attempts
```

#### **File Modified**: `src/detector.py` - Headless Compatibility

##### **Modified `show_results` Method (Line 426)**
```python
# Original:
def show_results(self, debugger, image, results):
    # ... processing code ...
    debugger.show_all_imgs(pause=self.pause)

# Modified:
def show_results(self, debugger, image, results):
    # ... processing code ...
    try:
        debugger.show_all_imgs(pause=self.pause)
    except:
        pass  # Skip display errors in headless mode
```

#### **File Modified**: `src/utils/debugger.py` - Display Error Handling

##### **Modified `show_all_imgs` Method (Lines 234-237)**
```python
# Original:
def show_all_imgs(self, pause=False, Time=0):
    if 1:
        for i, v in self.imgs.items():
            cv2.imshow('{}'.format(i), v)
        if not self.with_3d:
            cv2.waitKey(0 if pause else 1)

# Modified:
def show_all_imgs(self, pause=False, Time=0):
    if 1:
        try:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
            if not self.with_3d:
                cv2.waitKey(0 if pause else 1)
        except:
            # Skip display in headless mode
            pass
```

##### **Modified `show_img` Method (Line 50)**
```python
# Original:
def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
        cv2.waitKey()

# Modified:
def show_img(self, pause = False, imgId = 'default'):
    try:
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()
    except:
        pass  # Skip display in headless mode
```

### **4. Enhanced test output control**

#### **File Modified**: `src/opts.py` - Enhanced Options (Optional)

```python
# Added for better output control
self.parser.add_argument('--output_dir', default='../results',
                         help='directory to save output videos and JSON results')
self.parser.add_argument('--match_input_fps', action='store_true',
                         help='automatically match input video frame rate')
self.parser.add_argument('--match_input_resolution', action='store_true',
                         help='automatically match input video resolution')
```

### **Summary**
- ✅ Apply import fix in `mobilenet.py`
- ✅ Replace `demo.py` with enhanced version
- ✅ Add try-catch blocks in `detector.py` and `debugger.py`
- ✅ Add DCNv2 implementation files
- ✅ Add git tag: `git tag v1.0.0`

---

