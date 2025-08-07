from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import cv2
import json
import copy
import numpy as np
import glob
from opts import opts
from detector import Detector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv', 'wmv']  # Added wmv
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

# Monkey patch cv2.imshow to do nothing
def dummy_imshow(window_name, image):
    pass

def dummy_waitKey(delay=0):
    return -1

cv2.imshow = dummy_imshow
cv2.waitKey = dummy_waitKey

def get_video_info(video_path):
    """Get video properties like fps, width, height"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Video info - FPS: {fps}, Width: {width}, Height: {height}, Total frames: {total_frames}")
    return fps, width, height

def process_single_video(opt, video_path, output_dir, detector):
    """Process a single video file"""
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"{'='*60}")
    
    # Get video properties
    fps, width, height = get_video_info(video_path)
    if fps is None:
        print(f"Skipping {video_path} due to read error")
        return False
    
    # Use video properties if match flags are set
    if hasattr(opt, 'match_input_fps') and opt.match_input_fps:
        opt.save_framerate = fps
    if hasattr(opt, 'match_input_resolution') and opt.match_input_resolution:
        opt.video_w = width
        opt.video_h = height
    
    # Open video
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Setup output paths
    video_name = os.path.basename(video_path).split('.')[0]
    out_video_path = os.path.join(output_dir, f"{opt.exp_id}_{video_name}.mp4")
    out_json_path = os.path.join(output_dir, f"{opt.exp_id}_{video_name}_results.json")
    
    print(f"Output video: {out_video_path}")
    print(f"Output JSON: {out_json_path}")
    
    # Initialize output video writer
    out = None
    if opt.save_video:
        # Use the video dimensions (original or specified)
        out_width = opt.video_w if opt.resize_video else width
        out_height = opt.video_h if opt.resize_video else height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More compatible codec
        out = cv2.VideoWriter(out_video_path, fourcc, opt.save_framerate, (out_width, out_height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {out_video_path}")
            cam.release()
            return False
        
        print(f"Video writer initialized: {out_width}x{out_height} @ {opt.save_framerate} fps")
    
    # Reset detector tracking for new video
    if hasattr(detector, 'reset_tracking'):
        detector.reset_tracking()
    
    if opt.debug < 5:
        detector.pause = False
    
    cnt = 0
    results = {}
    processed_frames = 0
    
    print("Starting frame processing...")
    
    while True:
        ret_frame, img = cam.read()
        if not ret_frame or img is None:
            break
            
        cnt += 1
        
        # Skip initial frames if specified
        if opt.skip_first > 0 and cnt < opt.skip_first:
            continue
        
        # Resize if requested
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))
        
        # Process frame through detector
        try:
            ret = detector.run(img)
        except Exception as e:
            print(f"Error processing frame {cnt}: {str(e)}")
            continue
        
        processed_frames += 1
        
        # Log timing every 100 frames to avoid spam
        if cnt % 100 == 0:
            time_str = f'frame {cnt} |'
            for stat in time_stats:
                time_str += f' {stat} {ret[stat]:.3f}s |'
            print(time_str)
        
        # Store results
        results[cnt] = ret['results']
        
        # Save frame to output video
        if opt.save_video and out is not None:
            if 'generic' in ret and ret['generic'] is not None:
                # Use the processed frame with detections
                frame_to_save = ret['generic']
            else:
                # Fallback to original frame
                frame_to_save = img
            
            # Ensure frame dimensions match video writer expectations
            if frame_to_save.shape[:2] != (opt.video_h if opt.resize_video else height, 
                                          opt.video_w if opt.resize_video else width):
                target_height = opt.video_h if opt.resize_video else height
                target_width = opt.video_w if opt.resize_video else width
                frame_to_save = cv2.resize(frame_to_save, (target_width, target_height))
            
            out.write(frame_to_save)
    
    # Cleanup
    cam.release()
    if out is not None:
        out.release()
    
    # Save JSON results
    if opt.save_results and results:
        print(f'Saving {len(results)} frame results to {out_json_path}')
        json.dump(_to_list(copy.deepcopy(results)), open(out_json_path, 'w'))
    
    print(f"Completed processing: {video_name}")
    print(f"Processed {processed_frames} frames out of {cnt} total frames")
    print(f"Output saved to: {output_dir}")
    
    return True

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    
    # Create output directory if it doesn't exist
    output_dir = opt.output_dir if hasattr(opt, 'output_dir') else '../results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize detector once
    print("Initializing detector...")
    detector = Detector(opt)
    print("Detector initialized successfully")
    
    # Handle different input types
    if opt.demo == 'webcam':
        print("Webcam mode - processing single stream")
        # Original webcam code
        cam = cv2.VideoCapture(0)
        
        # Initialize output video for webcam
        out = None
        if opt.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(os.path.join(output_dir, f'{opt.exp_id}_webcam.mp4'),
                                fourcc, opt.save_framerate, (opt.video_w, opt.video_h))
        
        if opt.debug < 5:
            detector.pause = False
        
        cnt = 0
        results = {}
        
        while True:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, 'webcam')
            
            cnt += 1
            if cnt < opt.skip_first:
                continue
            
            if opt.resize_video:
                img = cv2.resize(img, (opt.video_w, opt.video_h))
            
            cv2.imshow('input', img)
            ret = detector.run(img)
            
            # Log timing
            time_str = f'frame {cnt} |'
            for stat in time_stats:
                time_str += f' {stat} {ret[stat]:.3f}s |'
            print(time_str)
            
            results[cnt] = ret['results']
            
            if opt.save_video and out is not None:
                if 'generic' in ret:
                    out.write(ret['generic'])
                else:
                    out.write(img)
            
            if cv2.waitKey(1) == 27:  # ESC key
                save_and_exit(opt, out, results, 'webcam')
                return
        
    elif os.path.isdir(opt.demo):
        # Process all videos in directory
        print(f"Scanning directory: {opt.demo}")
        video_files = []
        for ext in video_ext:
            pattern = os.path.join(opt.demo, f"*.{ext}")
            video_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(opt.demo, f"*.{ext.upper()}")
            video_files.extend(glob.glob(pattern))
        
        if not video_files:
            print(f"No video files found in {opt.demo}")
            print(f"Supported formats: {', '.join(video_ext)}")
            return
        
        print(f"Found {len(video_files)} video files:")
        for vf in sorted(video_files):
            print(f"  - {os.path.basename(vf)}")
        
        successful_count = 0
        for i, video_file in enumerate(sorted(video_files)):
            print(f"\n[{i+1}/{len(video_files)}] Processing: {os.path.basename(video_file)}")
            try:
                success = process_single_video(opt, video_file, output_dir, detector)
                if success:
                    successful_count += 1
                else:
                    print(f"Failed to process: {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Successfully processed: {successful_count}/{len(video_files)} videos")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
    elif os.path.isfile(opt.demo):
        # Single video file
        print(f"Processing single video file: {opt.demo}")
        success = process_single_video(opt, opt.demo, output_dir, detector)
        if success:
            print("Single video processing completed successfully")
        else:
            print("Failed to process the video file")
    else:
        # Handle image sequences (original functionality)
        print(f"Processing as image sequence: {opt.demo}")
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]
        
        if not image_names:
            print(f"No supported files found in: {opt.demo}")
            return
        
        # Initialize output video for image sequence
        out = None
        out_name = opt.demo[opt.demo.rfind('/') + 1:]
        if opt.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(os.path.join(output_dir, f'{opt.exp_id}_{out_name}.mp4'),
                                fourcc, opt.save_framerate, (opt.video_w, opt.video_h))
        
        if opt.debug < 5:
            detector.pause = False
        
        cnt = 0
        results = {}
        
        for img_path in image_names:
            cnt += 1
            if cnt < opt.skip_first:
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            if opt.resize_video:
                img = cv2.resize(img, (opt.video_w, opt.video_h))
            
            ret = detector.run(img)
            
            time_str = f'frame {cnt} |'
            for stat in time_stats:
                time_str += f' {stat} {ret[stat]:.3f}s |'
            print(time_str)
            
            results[cnt] = ret['results']
            
            if opt.save_video and out is not None:
                if 'generic' in ret:
                    out.write(ret['generic'])
                else:
                    out.write(img)
        
        save_and_exit(opt, out, results, out_name)

def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        output_dir = opt.output_dir if hasattr(opt, 'output_dir') else '../results'
        save_dir = os.path.join(output_dir, f'{opt.exp_id}_{out_name}_results.json')
        print('Saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)), open(save_dir, 'w'))
    
    if opt.save_video and out is not None:
        out.release()
    
    sys.exit(0)

def _to_list(results):
    """Convert numpy arrays to lists for JSON serialization"""
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)