#!/usr/bin/env python3
"""Convert nuScenes info files from old format (infos, metadata) to new format (data_list, metainfo).

MMDetection3D v2.0+ expects info files with 'data_list' and 'metainfo' keys,
but the converter creates 'infos' and 'metadata'.
"""

import pickle
from pathlib import Path


def convert_info_file(input_path: str, output_path: str):
    """Convert info file from old to new format and transform structure."""
    print(f"Converting {input_path}...")
    
    # Load old format
    with open(input_path, 'rb') as f:
        old_data = pickle.load(f)
    
    # Transform data_list to match MMDetection3D expected structure
    transformed_data_list = []
    for info in old_data['infos']:
        # MMDetection3D expects lidar_path nested under lidar_points
        transformed_info = info.copy()
        
        # Create lidar_points dict if it doesn't exist
        if 'lidar_points' not in transformed_info:
            transformed_info['lidar_points'] = {
                'lidar_path': transformed_info.pop('lidar_path', ''),
                'num_pts_feats': transformed_info.pop('num_features', 5),
                'lidar2ego': {
                    'translation': transformed_info.pop('lidar2ego_translation', [0, 0, 0]),
                    'rotation': transformed_info.pop('lidar2ego_rotation', [1, 0, 0, 0])
                }
            }
        
        # Transform lidar_sweeps if they exist
        if 'sweeps' in transformed_info:
            lidar_sweeps = []
            for sweep in transformed_info.pop('sweeps', []):
                sweep_data = {
                    'lidar_points': {
                        'lidar_path': sweep.get('data_path', ''),  # Changed from data_path to lidar_path
                        'lidar2ego': {
                            'translation': sweep.get('sensor2ego_translation', [0, 0, 0]),
                            'rotation': sweep.get('sensor2ego_rotation', [1, 0, 0, 0])
                        }
                    },
                    'ego2global': {
                        'translation': sweep.get('ego2global_translation', [0, 0, 0]),
                        'rotation': sweep.get('ego2global_rotation', [1, 0, 0, 0])
                    },
                    'timestamp': sweep.get('timestamp', 0)
                }
                if 'sensor2lidar_translation' in sweep:
                    sweep_data['lidar_points']['lidar2sensor'] = {
                        'translation': sweep['sensor2lidar_translation'],
                        'rotation': sweep.get('sensor2lidar_rotation', [1, 0, 0, 0])
                    }
                lidar_sweeps.append(sweep_data)
            transformed_info['lidar_sweeps'] = lidar_sweeps
        
        # Transform images/cams structure
        if 'cams' in transformed_info:
            transformed_info['images'] = {}
            for cam_name, cam_info in transformed_info.pop('cams', {}).items():
                transformed_info['images'][cam_name] = {
                    'img_path': cam_info.get('data_path', ''),
                    'cam2img': cam_info.get('cam_intrinsic', []),
                    'lidar2cam': cam_info.get('lidar2cam', [])
                }
        
        # Add ego2global transformation
        if 'ego2global_translation' in transformed_info:
            transformed_info['ego2global'] = {
                'translation': transformed_info.pop('ego2global_translation', [0, 0, 0]),
                'rotation': transformed_info.pop('ego2global_rotation', [1, 0, 0, 0])
            }
        
        # Convert gt_boxes, gt_names, etc. to instances format
        if 'gt_boxes' in transformed_info:
            instances = []
            gt_boxes = transformed_info.pop('gt_boxes', [])
            gt_names = transformed_info.pop('gt_names', [])
            gt_velocity = transformed_info.pop('gt_velocity', [])
            num_lidar_pts = transformed_info.pop('num_lidar_pts', [])
            num_radar_pts = transformed_info.pop('num_radar_pts', [])
            valid_flag = transformed_info.pop('valid_flag', [])
            
            # Map class names to indices using NuScenes METAINFO
            nuscenes_classes = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
            class_to_idx = {cls: idx for idx, cls in enumerate(nuscenes_classes)}
            
            for i in range(len(gt_boxes)):
                instance = {
                    'bbox_3d': gt_boxes[i].tolist() if hasattr(gt_boxes[i], 'tolist') else gt_boxes[i],
                    'bbox_label_3d': class_to_idx.get(gt_names[i] if isinstance(gt_names[i], str) else nuscenes_classes[gt_names[i]], 0),
                    'velocity': gt_velocity[i].tolist() if len(gt_velocity) > i and hasattr(gt_velocity[i], 'tolist') else gt_velocity[i] if len(gt_velocity) > i else [0.0, 0.0],
                    'num_lidar_pts': int(num_lidar_pts[i]) if len(num_lidar_pts) > i else 0,
                    'num_radar_pts': int(num_radar_pts[i]) if len(num_radar_pts) > i else 0,
                }
                instances.append(instance)
            
            transformed_info['instances'] = instances
        
        transformed_data_list.append(transformed_info)
    
    # Convert to new format
    new_data = {
        'data_list': transformed_data_list,
        'metainfo': old_data['metadata']
    }
    
    # Save new format
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print(f"✅ Converted {len(new_data['data_list'])} samples to {output_path}")


def main():
    """Convert train and val info files."""
    data_root = Path('data/nuscenes/v1.0-mini')
    
    # Convert train
    train_input = data_root / 'nuscenes_infos_train.pkl'
    train_output = data_root / 'nuscenes_infos_train.pkl'
    convert_info_file(train_input, train_output)
    
    # Convert val
    val_input = data_root / 'nuscenes_infos_val.pkl'
    val_output = data_root / 'nuscenes_infos_val.pkl'
    convert_info_file(val_input, val_output)
    
    print("\n✅ All info files converted to MMDetection3D v2.0 format!")


if __name__ == '__main__':
    main()

