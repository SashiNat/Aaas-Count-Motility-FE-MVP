#!/usr/bin/env python3
"""
Sperm Motility Analysis Script - Version 1
Processes CenterTrack JSON results to calculate speeds and classify according to WHO 2021 standards
Auto-detects video properties (FPS, resolution) from input files
"""

import json
import numpy as np
import cv2
import os
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class SpermAnalyzer:
    def __init__(self, pixels_per_micrometer=1.0):
        self.pixels_per_micrometer = pixels_per_micrometer
        self.fps = None
        self.video_width = None
        self.video_height = None
        
        # WHO 2021 Standards (micrometers per second)
        self.WHO_THRESHOLDS = {
            'rapid_progressive': 25.0,    # >= 25 μm/s
            'slow_progressive': 5.0,      # 5-24.9 μm/s  
            'non_progressive': 0.5,       # 0.5-4.9 μm/s (some movement)
            'immotile': 0.0               # < 0.5 μm/s
        }
        
        # Color coding for visualization (BGR format for OpenCV)
        self.COLORS = {
            'RP': (0, 255, 0),      # Green - Rapid Progressive
            'SP': (0, 255, 255),    # Yellow - Slow Progressive  
            'NP': (255, 0, 0),      # Blue - Non-Progressive
            'IM': (128, 128, 128),  # Gray - Immotile
            'UNKNOWN': (255, 255, 255)  # White - Unknown
        }
    
    def extract_video_properties(self, video_file):
        """Extract FPS, width, height from video file"""
        if not os.path.exists(video_file):
            print(f"Warning: Video file {video_file} not found")
            return None, None, None
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            return None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        print(f"📹 Video Properties:")
        print(f"  File: {os.path.basename(video_file)}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame Rate: {fps:.2f} fps")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        return fps, width, height
    
    def extract_frame_range_from_json(self, json_file):
        """Extract frame range and video properties from JSON results"""
        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} not found")
            return None, None, None
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("Error: JSON file is empty")
            return None, None, None
        
        # Get frame numbers
        frame_numbers = [int(frame_num) for frame_num in data.keys()]
        min_frame = min(frame_numbers)
        max_frame = max(frame_numbers)
        total_frames = len(frame_numbers)
        
        # Try to extract resolution from first detection with bbox
        width, height = None, None
        for frame_data in data.values():
            for detection in frame_data:
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    # Estimate resolution from bbox coordinates (rough estimate)
                    max_x = max(bbox[0], bbox[2])
                    max_y = max(bbox[1], bbox[3])
                    if width is None or max_x > width:
                        width = max_x
                    if height is None or max_y > height:
                        height = max_y
        
        # Add some padding to estimated resolution
        if width and height:
            width = int(width * 1.1)  # Add 10% padding
            height = int(height * 1.1)
        
        print(f"📊 JSON Analysis:")
        print(f"  File: {os.path.basename(json_file)}")
        print(f"  Frame Range: {min_frame} to {max_frame}")
        print(f"  Processed Frames: {total_frames}")
        if width and height:
            print(f"  Estimated Resolution: {width}x{height} (from bounding boxes)")
        
        return min_frame, max_frame, total_frames
    
    def auto_detect_fps(self, json_file, video_file=None):
        """Auto-detect FPS from video file or estimate from JSON"""
        fps = None
        
        # First try to get from video file
        if video_file:
            video_fps, _, _ = self.extract_video_properties(video_file)
            if video_fps and video_fps > 0:
                fps = video_fps
                print(f"✅ Using FPS from video file: {fps}")
        
        # If no video file or couldn't read FPS, estimate from JSON
        if fps is None:
            min_frame, max_frame, total_frames = self.extract_frame_range_from_json(json_file)
            if min_frame is not None and max_frame is not None:
                # Assume continuous frames and common frame rates
                frame_span = max_frame - min_frame + 1
                if frame_span == total_frames:
                    # Continuous frames - guess frame rate based on common values
                    if total_frames > 1000:
                        fps = 30.0  # Long video, likely 30fps
                    elif total_frames > 300:
                        fps = 25.0  # Medium video, likely 25fps
                    else:
                        fps = 30.0  # Short video, default to 30fps
                else:
                    # Non-continuous frames - estimate based on processing time
                    fps = 30.0  # Default assumption
                
                print(f"⚠️  Estimated FPS from JSON analysis: {fps} (frames: {total_frames})")
        
        if fps is None:
            fps = 30.0  # Final fallback
            print(f"⚠️  Using default FPS: {fps}")
        
        return fps
    
    def auto_detect_resolution(self, json_file, video_file=None):
        """Auto-detect video resolution from video file or JSON"""
        width, height = None, None
        
        # First try to get from video file
        if video_file:
            _, video_width, video_height = self.extract_video_properties(video_file)
            if video_width and video_height:
                width, height = video_width, video_height
                print(f"✅ Using resolution from video file: {width}x{height}")
        
        # If no video file or couldn't read resolution, estimate from JSON
        if width is None or height is None:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            max_x, max_y = 0, 0
            bbox_count = 0
            
            for frame_data in data.values():
                for detection in frame_data:
                    if 'bbox' in detection:
                        bbox = detection['bbox']
                        max_x = max(max_x, bbox[0], bbox[2])
                        max_y = max(max_y, bbox[1], bbox[3])
                        bbox_count += 1
            
            if bbox_count > 0:
                # Add padding to estimated resolution
                width = int(max_x * 1.2)  # 20% padding
                height = int(max_y * 1.2)
                print(f"⚠️  Estimated resolution from JSON bounding boxes: {width}x{height}")
            else:
                # Default resolution if no bounding boxes found
                width, height = 640, 480
                print(f"⚠️  Using default resolution: {width}x{height}")
        
        return width, height

    def calculate_pixel_to_micrometer_ratio(self, magnification=400, 
                                          sensor_width_mm=4.8, sensor_height_mm=3.6,
                                          video_width=640, video_height=384):
        """Calculate pixel to micrometer conversion ratio from microscope specs"""
        
        # Calculate field of view in mm
        fov_width_mm = sensor_width_mm / magnification
        fov_height_mm = sensor_height_mm / magnification
        
        # Convert to micrometers
        fov_width_um = fov_width_mm * 1000
        fov_height_um = fov_height_mm * 1000
        
        # Calculate pixels per micrometer
        pixels_per_um_x = video_width / fov_width_um
        pixels_per_um_y = video_height / fov_height_um
        
        # Use average
        pixels_per_um = (pixels_per_um_x + pixels_per_um_y) / 2
        
        print(f"Microscope Analysis:")
        print(f"  Magnification: {magnification}x")
        print(f"  Video Resolution: {video_width}x{video_height}")
        print(f"  Field of View: {fov_width_um:.1f} x {fov_height_um:.1f} μm")
        print(f"  Pixels per micrometer: {pixels_per_um:.2f}")
        
        return pixels_per_um
    
    def extract_tracks_from_json(self, json_file):
        """Extract tracking data from CenterTrack JSON results"""
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Organize data by tracking ID
        tracks = defaultdict(list)
        
        for frame_num, detections in data.items():
            frame_num = int(frame_num)
            
            for detection in detections:
                if 'tracking_id' in detection and 'bbox' in detection:
                    track_id = detection['tracking_id']
                    
                    # Calculate center point from bbox
                    bbox = detection['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    tracks[track_id].append({
                        'frame': frame_num,
                        'x': center_x,
                        'y': center_y,
                        'bbox': bbox,
                        'score': detection.get('score', 1.0)
                    })
        
        return tracks
    
    def calculate_speeds(self, tracks):
        """Calculate speed for each track"""
        
        track_speeds = {}
        track_classifications = {}
        
        for track_id, positions in tracks.items():
            if len(positions) < 2:
                # Not enough points to calculate speed
                track_speeds[track_id] = {'avg_speed': 0.0, 'max_speed': 0.0, 'track_length': len(positions), 'duration': 0}
                track_classifications[track_id] = 'IM'
                continue
            
            # Sort by frame number
            positions = sorted(positions, key=lambda x: x['frame'])
            
            # Calculate instantaneous speeds
            speeds = []
            
            for i in range(1, len(positions)):
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                
                # Calculate distance in pixels
                dx = curr_pos['x'] - prev_pos['x']
                dy = curr_pos['y'] - prev_pos['y']
                distance_pixels = np.sqrt(dx**2 + dy**2)
                
                # Calculate time difference (assuming consecutive frames)
                frame_diff = curr_pos['frame'] - prev_pos['frame']
                time_diff = frame_diff / self.fps  # seconds
                
                if time_diff > 0:
                    # Speed in pixels per second
                    speed_pixels_per_sec = distance_pixels / time_diff
                    
                    # Convert to micrometers per second
                    speed_um_per_sec = speed_pixels_per_sec / self.pixels_per_micrometer
                    
                    speeds.append(speed_um_per_sec)
            
            # Use average speed for classification
            if speeds:
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                
                # Store both average and max speed
                track_speeds[track_id] = {
                    'avg_speed': avg_speed,
                    'max_speed': max_speed,
                    'track_length': len(positions),
                    'duration': (positions[-1]['frame'] - positions[0]['frame']) / self.fps
                }
                
                # Classify based on average speed
                classification = self.classify_sperm(avg_speed)
                track_classifications[track_id] = classification
            else:
                track_speeds[track_id] = {'avg_speed': 0.0, 'max_speed': 0.0, 'track_length': len(positions), 'duration': 0}
                track_classifications[track_id] = 'IM'
        
        return track_speeds, track_classifications
    
    def classify_sperm(self, speed_um_per_sec):
        """Classify sperm based on WHO 2021 standards"""
        
        if speed_um_per_sec >= self.WHO_THRESHOLDS['rapid_progressive']:
            return 'RP'
        elif speed_um_per_sec >= self.WHO_THRESHOLDS['slow_progressive']:
            return 'SP'
        elif speed_um_per_sec >= self.WHO_THRESHOLDS['non_progressive']:
            return 'NP'
        else:
            return 'IM'
    
    def generate_who_report(self, classifications):
        """Generate WHO standard report"""
        
        # Count classifications
        counts = Counter(classifications.values())
        total = sum(counts.values())
        
        if total == 0:
            return "No tracks found for analysis"
        
        # Calculate percentages
        rp_count = counts.get('RP', 0)
        sp_count = counts.get('SP', 0)  
        np_count = counts.get('NP', 0)
        im_count = counts.get('IM', 0)
        
        rp_pct = (rp_count / total) * 100
        sp_pct = (sp_count / total) * 100
        np_pct = (np_count / total) * 100
        im_pct = (im_count / total) * 100
        
        total_motile_pct = rp_pct + sp_pct + np_pct
        total_progressive_pct = rp_pct + sp_pct
        
        report = f"""
WHO 2021 Sperm Motility Analysis Report
{'='*50}
Total Sperm Analyzed: {total}

Individual Categories:
  1. Rapid Progressive (RP): {rp_count} ({rp_pct:.1f}%)
  2. Slow Progressive (SP): {sp_count} ({sp_pct:.1f}%)
  3. Non-Progressive (NP): {np_count} ({np_pct:.1f}%)
  4. Immotile (IM): {im_count} ({im_pct:.1f}%)

WHO Standard Categories:
  1. Total Motile (RP+SP+NP): {rp_count + sp_count + np_count} ({total_motile_pct:.1f}%)
  2. Total Progressive (RP+SP): {rp_count + sp_count} ({total_progressive_pct:.1f}%)
  3. Rapid Progressive (RP): {rp_count} ({rp_pct:.1f}%)
  4. Slow Progressive (SP): {sp_count} ({sp_pct:.1f}%)
  5. Non-Progressive (NP): {np_count} ({np_pct:.1f}%)
  6. Immotile (IM): {im_count} ({im_pct:.1f}%)

WHO Reference Values (Normal):
  - Total Motile: ≥40%
  - Progressive Motile: ≥32%
  - Rapid Progressive: No specific threshold
"""
        return report
    
    def create_colored_video(self, original_video_path, json_file, output_path, classifications, tracks):
        """Create color-coded video based on classifications"""
        
        # Open original video
        cap = cv2.VideoCapture(original_video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {original_video_path}")
            return
        
        # Get video properties (use detected values as fallback)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.video_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.video_height
        
        print(f"📹 Creating color-coded video:")
        print(f"  Input: {os.path.basename(original_video_path)}")
        print(f"  Output: {os.path.basename(output_path)}")
        print(f"  Properties: {width}x{height} @ {fps:.2f} fps")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load detection data
        with open(json_file, 'r') as f:
            detection_data = json.load(f)
        
        frame_num = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Draw detections for this frame
            if str(frame_num) in detection_data:
                detections = detection_data[str(frame_num)]
                
                for detection in detections:
                    if 'tracking_id' in detection and 'bbox' in detection:
                        track_id = detection['tracking_id']
                        bbox = detection['bbox']
                        
                        # Get classification and color
                        classification = classifications.get(track_id, 'UNKNOWN')
                        color = self.COLORS[classification]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    color, 1)
                        
                        # Draw classification text
                        text = f"{classification}{track_id}"
                        cv2.putText(frame, text, 
                                  (int(bbox[0]), int(bbox[1]) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                processed_frames += 1
            
            out.write(frame)
            
            if frame_num % 100 == 0:
                print(f"  Processed {frame_num} frames...")
        
        cap.release()
        out.release()
        
        print(f"✅ Color-coded video saved: {output_path}")
        print(f"  Total frames: {frame_num}")
        print(f"  Frames with detections: {processed_frames}")
        
        # Add legend info
        print(f"\n🎨 Color Legend:")
        print(f"  🟢 Green (RP): Rapid Progressive (≥25 μm/s)")
        print(f"  🟡 Yellow (SP): Slow Progressive (5-24.9 μm/s)")
        print(f"  🔵 Blue (NP): Non-Progressive (0.5-4.9 μm/s)")
        print(f"  ⚫ Gray (IM): Immotile (<0.5 μm/s)")
    
    def analyze_video(self, json_file, original_video_path=None, output_dir="./analysis_results"):
        """Complete analysis pipeline with auto-detection of video properties"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract clean filename from JSON file for output naming
        json_basename = os.path.splitext(os.path.basename(json_file))[0]
        
        # Remove common prefixes to get clean identifier
        clean_basename = self._clean_filename(json_basename)
        
        print("🧬 Starting Sperm Motility Analysis...")
        print("="*60)
        print(f"📁 Input file: {os.path.basename(json_file)}")
        print(f"📁 Output prefix: {clean_basename}")
        
        # Auto-detect video properties
        print("\n1️⃣ Auto-detecting video properties...")
        self.fps = self.auto_detect_fps(json_file, original_video_path)
        self.video_width, self.video_height = self.auto_detect_resolution(json_file, original_video_path)
        
        print(f"\n📋 Analysis Parameters:")
        print(f"  FPS: {self.fps}")
        print(f"  Resolution: {self.video_width}x{self.video_height}")
        print(f"  Pixels per micrometer: {self.pixels_per_micrometer}")
        
        # Extract tracks
        print(f"\n2️⃣ Extracting tracks from JSON...")
        tracks = self.extract_tracks_from_json(json_file)
        print(f"Found {len(tracks)} tracks")
        
        if len(tracks) == 0:
            print("❌ No tracks found in JSON file!")
            return None, "No tracks found for analysis"
        
        # Calculate speeds
        print(f"\n3️⃣ Calculating speeds...")
        speeds, classifications = self.calculate_speeds(tracks)
        
        # Show some statistics
        valid_tracks = sum(1 for speed_data in speeds.values() 
                          if isinstance(speed_data, dict) and speed_data.get('avg_speed', 0) > 0)
        print(f"Valid tracks with movement: {valid_tracks}/{len(tracks)}")
        
        # Generate report
        print(f"\n4️⃣ Generating WHO report...")
        report = self.generate_who_report(classifications)
        print(report)
        
        # Save detailed results with clean filename
        detailed_file = os.path.join(output_dir, f"{clean_basename}_detailed_analysis.json")
        detailed_results = {
            'analysis_info': {
                'input_json_file': json_file,
                'input_video_file': original_video_path,
                'original_filename': json_basename,
                'output_prefix': clean_basename,
                'analysis_timestamp': self._get_timestamp(),
                'fps': self.fps,
                'resolution': {'width': self.video_width, 'height': self.video_height},
                'pixels_per_micrometer': self.pixels_per_micrometer,
                'who_thresholds': self.WHO_THRESHOLDS,
                'total_tracks': len(tracks),
                'valid_tracks': valid_tracks
            },
            'tracks': {}
        }
        
        for track_id in tracks.keys():
            detailed_results['tracks'][track_id] = {
                'classification': classifications[track_id],
                'speed_data': speeds[track_id],
                'positions': tracks[track_id]
            }
        
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save report with clean filename
        report_file = os.path.join(output_dir, f"{clean_basename}_WHO_motility_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"Analysis of: {os.path.basename(json_file)}\n")
            f.write(f"Sample ID: {clean_basename}\n")
            f.write(f"Generated: {self._get_timestamp()}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        # Create color-coded video if original video provided
        if original_video_path and os.path.exists(original_video_path):
            print(f"\n5️⃣ Creating color-coded video...")
            output_video = os.path.join(output_dir, f"{clean_basename}_classified_sperm_video.mp4")
            self.create_colored_video(original_video_path, json_file, output_video, classifications, tracks)
        else:
            print(f"\n5️⃣ Skipping video creation (no input video provided)")
        
        print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
        print(f"📁 Output files:")
        print(f"  - {clean_basename}_WHO_motility_report.txt")
        print(f"  - {clean_basename}_detailed_analysis.json")
        if original_video_path:
            print(f"  - {clean_basename}_classified_sperm_video.mp4")
        
        return detailed_results, report
    
    def _clean_filename(self, filename):
        """
        Clean filename by removing common prefixes and keeping core identifier
        Examples:
        'mot17_half_sc_CR_001_results' -> 'CR_001'
        'centertrack_CR_002_results' -> 'CR_002'
        'exp_id_sample_123_results' -> 'sample_123'
        """
        
        # Common prefixes to remove
        prefixes_to_remove = [
            'mot17_half_sc_',
            'centertrack_',
            'exp_id_',
            'tracking_',
            'results_'
        ]
        
        # Common suffixes to remove
        suffixes_to_remove = [
            '_results',
            '_tracking',
            '_output'
        ]
        
        cleaned = filename
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):]
                break
        
        # Remove suffixes
        for suffix in suffixes_to_remove:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[:-len(suffix)]
                break
        
        # If still empty or too generic, use original
        if not cleaned or len(cleaned) < 3:
            cleaned = filename
        
        return cleaned
    
    def _get_timestamp(self):
        """Get current timestamp for record keeping"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description='Sperm Motility Analysis - WHO 2021 Standards')
    parser.add_argument('--json_file', required=True, help='Path to CenterTrack JSON results')
    parser.add_argument('--video_file', help='Path to original video file (optional, for color-coded output)')
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory')
    parser.add_argument('--pixels_per_um', type=float, default=1.0, help='Pixels per micrometer conversion')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.json_file):
        print(f"❌ Error: JSON file not found: {args.json_file}")
        return
    
    if args.video_file and not os.path.exists(args.video_file):
        print(f"⚠️  Warning: Video file not found: {args.video_file}")
        args.video_file = None
    
    # Initialize analyzer
    print("🔬 Initializing Sperm Motility Analyzer...")
    analyzer = SpermAnalyzer(pixels_per_micrometer=args.pixels_per_um)
    
    # Run analysis
    try:
        results, report = analyzer.analyze_video(
            json_file=args.json_file,
            original_video_path=args.video_file,
            output_dir=args.output_dir
        )
        
        if results:
            print("\n🎉 Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()