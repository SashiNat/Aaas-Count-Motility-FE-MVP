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

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  
  # Force minimal debug settings
  opt.debug = 0
  opt.not_show_bbox = True
  opt.not_show_number = True
  opt.not_show_txt = True
  
  detector = Detector(opt)
  
  # Open video
  cap = cv2.VideoCapture(opt.demo)
  if not cap.isOpened():
      print(f"Error: Could not open video {opt.demo}")
      return
  
  print(f"Video opened: {opt.demo}")
  print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
  
  results = {}
  cnt = 0
  successful_frames = 0
  
  # Create results directory
  os.makedirs('../results', exist_ok=True)
  
  while True:
      ret, img = cap.read()
      if not ret or img is None:
          break
      
      cnt += 1
      if cnt > 20:  # Process only 20 frames
          break
          
      print(f"Processing frame {cnt}...")
      
      try:
          # Run detection/tracking
          result = detector.run(img, {})
          results[cnt] = result['results']
          successful_frames += 1
          
          # Print basic info
          print(f"  Detections: {len(result['results'])}")
          if result['results']:
              for i, det in enumerate(result['results'][:2]):
                  bbox = det.get('bbox', [0,0,0,0])
                  track_id = det.get('tracking_id', 'N/A')
                  score = det.get('score', 0)
                  print(f"    {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}], track_id={track_id}, score={score:.2f}")
          
      except Exception as e:
          print(f"Error on frame {cnt}: {e}")
          continue
  
  cap.release()
  
  # Save results
  if results:
      out_name = opt.demo.split('/')[-1]
      save_path = f'../results/{opt.exp_id}_{out_name}_results.json'
      print(f'Saving {successful_frames} results to {save_path}')
      
      # Convert numpy arrays to lists for JSON serialization
      for img_id in results:
          for t in range(len(results[img_id])):
              for k in results[img_id][t]:
                  if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                      results[img_id][t][k] = results[img_id][t][k].tolist()
      
      json.dump(results, open(save_path, 'w'))
      print("Done!")

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
