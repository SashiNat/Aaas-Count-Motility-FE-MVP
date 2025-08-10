#!/usr/bin/env python3
"""
Convert Sperm MOT format (from CVAT) to CenterTrack COCO format
Adapted from official CenterTrack convert_mot_to_coco.py
"""

import os
import numpy as np
import json
import cv2
from pathlib import Path
import argparse

def convert_sperm_mot_to_coco(mot_data_path, output_path, train_split=0.8):
    """
    Convert MOT format sperm data to CenterTrack COCO format
    
    Args:
        mot_data_path: Path to MOT export directory (contains img1/ and gt/)
        output_path: Output directory for CenterTrack data
        train_split: Training split ratio (0.8 = 80% train, 20% val)
    """
    
    mot_data_path = Path(mot_data_path)
    output_path = Path(output_path)
    
    # Create output directories
    annotations_dir = output_path / "annotations"
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    for dir_path in [annotations_dir, train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Input paths
    img_path = mot_data_path / 'img1'
    gt_path = mot_data_path / 'gt' / 'gt.txt'
    
    print(f"📁 MOT data path: {mot_data_path}")
    print(f"📁 Images path: {img_path}")
    print(f"📁 Annotations path: {gt_path}")
    print(f"📁 Output path: {output_path}")
    
    # Check if paths exist
    if not img_path.exists():
        raise FileNotFoundError(f"Images directory not found: {img_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    # Get all image files
    images = [f for f in os.listdir(img_path) if f.endswith('.jpg') or f.endswith('.png')]
    images.sort()
    num_images = len(images)
    
    if num_images == 0:
        raise ValueError(f"No images found in {img_path}")
    
    print(f"📊 Found {num_images} images")
    
    # Get image dimensions from first image
    sample_img_path = img_path / images[0]
    sample_img = cv2.imread(str(sample_img_path))
    if sample_img is None:
        raise ValueError(f"Could not read sample image: {sample_img_path}")
    
    height, width = sample_img.shape[:2]
    print(f"📊 Image dimensions: {width}x{height}")
    
    # Load annotations
    print("📖 Loading MOT annotations...")
    anns = np.loadtxt(gt_path, dtype=np.float32, delimiter=',')
    print(f"📊 Loaded {len(anns)} annotations")
    print(f"📊 Frame range: {int(anns[:, 0].min())} to {int(anns[:, 0].max())}")
    print(f"📊 Unique tracks: {len(np.unique(anns[:, 1]))}")
    
    # Define splits
    SPLITS = ['train', 'val']
    HALF_VIDEO = True
    
    for split in SPLITS:
        print(f"\n🔄 Processing {split} split...")
        
        out_path = annotations_dir / f'{split}.json'
        out = {
            'images': [], 
            'annotations': [], 
            'categories': [{'id': 1, 'name': 'sperm'}],
            'videos': [{
                'id': 1,
                'file_name': 'sperm_video',
                'width': width,
                'height': height,
                'length': num_images
            }]
        }
        
        # Determine image range for split
        if split == 'train':
            image_range = [0, int(num_images * train_split) - 1]
            target_dir = train_dir
        else:  # val
            image_range = [int(num_images * train_split), num_images - 1]
            target_dir = val_dir
        
        print(f"   Frame range: {image_range[0]} to {image_range[1]}")
        
        image_cnt = 0
        ann_cnt = 0
        
        # Process images
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            
            # Copy image to target directory with sequential naming
            src_img_path = img_path / images[i]
            frame_id = i - image_range[0]
            new_filename = f"{frame_id:06d}.jpg"
            dst_img_path = target_dir / new_filename
            
            # Copy image
            import shutil
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create image info
            image_info = {
                'file_name': new_filename,
                'id': image_cnt + 1,
                'frame_id': frame_id,
                'prev_image_id': image_cnt if image_cnt > 0 else -1,
                'next_image_id': image_cnt + 2 if i < image_range[1] else -1,
                'video_id': 1,
                'width': width,
                'height': height
            }
            out['images'].append(image_info)
            image_cnt += 1
        
        # Process annotations
        print(f"   Processing annotations for {len(out['images'])} images...")
        
        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0]) - 1  # MOT frames are 1-indexed
            
            # Check if frame is in current split
            if frame_id < image_range[0] or frame_id > image_range[1]:
                continue
            
            track_id = int(anns[i][1])
            
            # MOT format: frame, id, x, y, w, h, conf, class, visibility
            bbox = anns[i][2:6].tolist()  # [x, y, w, h]
            conf = float(anns[i][6]) if len(anns[i]) > 6 else 1.0
            
            # Skip low confidence detections (optional)
            if conf < 0.1:
                continue
            
            ann_cnt += 1
            
            # Convert frame_id to image_id in current split
            split_frame_id = frame_id - image_range[0]
            image_id = split_frame_id + 1
            
            ann = {
                'id': ann_cnt,
                'category_id': 1,  # Always sperm
                'image_id': image_id,
                'track_id': track_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],  # width * height
                'iscrowd': 0,
                'conf': conf
            }
            out['annotations'].append(ann)
        
        print(f"   ✅ {split}: {len(out['images'])} images, {len(out['annotations'])} annotations")
        
        # Update video length for this split
        out['videos'][0]['length'] = len(out['images'])
        
        # Save annotations
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        
        print(f"   💾 Saved: {out_path}")
    
    # Create summary
    train_json = annotations_dir / 'train.json'
    val_json = annotations_dir / 'val.json'
    
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    # Calculate statistics
    all_track_ids = set()
    for ann in train_data['annotations'] + val_data['annotations']:
        all_track_ids.add(ann['track_id'])
    
    summary = {
        "dataset_name": "sperm_tracking_mot",
        "conversion_source": "MOT format from CVAT",
        "train_split": train_split,
        "statistics": {
            "total_images": num_images,
            "train_images": len(train_data['images']),
            "val_images": len(val_data['images']),
            "train_annotations": len(train_data['annotations']),
            "val_annotations": len(val_data['annotations']),
            "unique_tracks": len(all_track_ids),
            "total_annotations": len(train_data['annotations']) + len(val_data['annotations']),
            "avg_annotations_per_frame": (len(train_data['annotations']) + len(val_data['annotations'])) / num_images
        },
        "video_info": {
            "width": width,
            "height": height,
            "total_frames": num_images
        },
        "paths": {
            "train_annotations": str(train_json),
            "val_annotations": str(val_json),
            "train_images": str(train_dir),
            "val_images": str(val_dir)
        }
    }
    
    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Conversion completed successfully!")
    print(f"📊 Final Statistics:")
    print(f"   Total images: {num_images}")
    print(f"   Training: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"   Validation: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"   Unique tracks: {len(all_track_ids)}")
    print(f"   Avg annotations per frame: {summary['statistics']['avg_annotations_per_frame']:.1f}")
    
    print(f"\n📁 Output Structure:")
    print(f"   {output_path}/")
    print(f"   ├── annotations/")
    print(f"   │   ├── train.json")
    print(f"   │   └── val.json")
    print(f"   ├── train/ ({len(train_data['images'])} images)")
    print(f"   ├── val/ ({len(val_data['images'])} images)")
    print(f"   └── dataset_summary.json")
    
    print(f"\n🚀 Ready for CenterTrack training!")
    print(f"📋 Training command:")
    print(f"cd $CenterTrack_ROOT/src")
    print(f"python main.py tracking \\")
    print(f"  --exp_id sperm_tracking_mot \\")
    print(f"  --dataset custom \\")
    print(f"  --custom_dataset_ann_path {train_json} \\")
    print(f"  --custom_dataset_img_path {output_path}/ \\")
    print(f"  --input_h {height} --input_w {width} \\")
    print(f"  --num_classes 1 \\")
    print(f"  --batch_size 8 \\")
    print(f"  --num_epochs 70 \\")
    print(f"  --lr 2.5e-4 \\")
    print(f"  --load_model ../models/mot17_half_sc.pth \\")
    print(f"  --pre_hm --ltrb_amodal --same_aug \\")
    print(f"  --hm_disturb 0.02 --lost_disturb 0.2 --fp_disturb 0.05 \\")
    print(f"  --dense_hp --max_objs {max(100, int(summary['statistics']['avg_annotations_per_frame'] * 1.5))} \\")
    print(f"  --gpus 0")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Convert Sperm MOT format to CenterTrack COCO format')
    parser.add_argument('mot_data_path', help='Path to MOT export directory (contains img1/ and gt/)')
    parser.add_argument('--output_path', default=None, help='Output directory (default: $CenterTrack_ROOT/data/sperm)')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        centertrack_root = os.environ.get('CenterTrack_ROOT', '/home/azureuser/CenterTrack')
        args.output_path = os.path.join(centertrack_root, 'data', 'sperm')
    
    # Convert dataset
    summary = convert_sperm_mot_to_coco(
        args.mot_data_path,
        args.output_path,
        args.train_split
    )
    
    print(f"\n🎯 Next steps:")
    print(f"1. Verify the converted data structure")
    print(f"2. Start training with the command above")
    print(f"3. Monitor training progress")
    
    return summary

if __name__ == "__main__":
    main()

# Example usage:
# python convert_sperm_mot_to_coco.py /path/to/mot_export --output_path $CenterTrack_ROOT/data/sperm