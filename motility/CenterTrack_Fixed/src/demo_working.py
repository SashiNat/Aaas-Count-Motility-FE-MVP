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
from opts import opts
from detector import Detector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv', 'wmv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    
    if not cam.isOpened():
        print(f"Error: Could not open video {opt.demo}")
        return
        
    print(f"Video opened successfully: {opt.demo}")
    print(f"Video properties: {int(cam.get(cv2.CAP_PROP_FRAME_COUNT))} frames, "
          f"{cam.get(cv2.CAP_PROP_FPS):.2f} FPS")
  else:
    is_video = False
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

  # Initialize output
  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('Processing:', out_name)
  
  # Create results directory
  os.makedirs('../results', exist_ok=True)
  
  if opt.debug < 5:
    detector.pause = False
  
  cnt = 0
  results = {}
  successful_frames = 0

  while True:
      if is_video:
        ret, img = cam.read()
        if not ret:
          print(f"Finished reading video at frame {cnt}")
          break
        if img is None:
          print(f"Warning: Got None image at frame {cnt}")
          continue
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
          if img is None:
            print(f"Failed to read image {image_names[cnt]}")
            cnt += 1
            continue
        else:
          break
      
      cnt += 1

      # Skip frames if needed
      if cnt < opt.skip_first:
        continue

      # Resize if needed
      if opt.resize_video:
        img = cv2.resize(img, (opt.video_w, opt.video_h))

      print(f"Processing frame {cnt}, shape: {img.shape}")

      try:
        # Call detector.run with image array and empty meta dict
        ret = detector.run(img, {})
        successful_frames += 1
        
        # Log timing
        time_str = f'frame {cnt} |'
        for stat in time_stats:
          if stat in ret:
            time_str += f'{stat} {ret[stat]:.3f}s |'
        print(time_str)

        # Store results
        results[cnt] = ret['results']
        
        # Print detection info
        if ret['results']:
          print(f"  Found {len(ret['results'])} detections")
          for i, det in enumerate(ret['results'][:3]):  # Show first 3
            bbox = det.get('bbox', [0,0,0,0])
            track_id = det.get('tracking_id', 'N/A')
            score = det.get('score', 0)
            print(f"    Det {i}: bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}], "
                  f"track_id={track_id}, score={score:.3f}")
        else:
          print("  No detections")

      except Exception as e:
        print(f"Error processing frame {cnt}: {e}")
        import traceback
        traceback.print_exc()
        continue

      # Process limited frames for testing
      if cnt >= 20:  # Process 20 frames
        print(f"Processed {successful_frames} frames successfully!")
        break

  if is_video:
    cam.release()

  # Save results
  if opt.save_results and results:
    save_dir = f'../results/{opt.exp_id}_{out_name}_results.json'
    print(f'Saving results to {save_dir}')
    json.dump(_to_list(copy.deepcopy(results)), open(save_dir, 'w'))
    print(f"Results saved! Processed {successful_frames} frames total.")

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
