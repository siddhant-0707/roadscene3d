"""nuScenes dataset loader for MMDetection3D.

This module handles loading and preprocessing nuScenes mini dataset
and converting it to MMDetection3D format.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, Box
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    logger.warning("nuscenes-devkit not available. Install with: pip install nuscenes-devkit")


class NuScenesDataset:
    """Dataset loader for nuScenes dataset converted to MMDetection3D format."""
    
    # Mapping from nuScenes category names to our class indices
    # nuScenes uses hierarchical categories like "vehicle.car", "human.pedestrian.adult"
    NUSCENES_TO_CLASS = {
        # Vehicles
        'vehicle.car': 0,
        'vehicle.truck': 1,
        'vehicle.bus': 2,
        'vehicle.trailer': 3,
        'vehicle.construction': 3,  # Map to trailer
        'vehicle.emergency': 0,  # Map to car
        # Pedestrians
        'human.pedestrian.adult': 4,
        'human.pedestrian.child': 4,
        'human.pedestrian.wheelchair': 4,
        'human.pedestrian.stroller': 4,
        'human.pedestrian.personal_mobility': 4,
        'human.pedestrian.police_officer': 4,
        'human.pedestrian.construction_worker': 4,
        # Cyclists
        'vehicle.bicycle': 5,
        'vehicle.motorcycle': 6,
        # Other
        'movable_object.trafficcone': 7,
        'movable_object.barrier': 8,
        'static_object.bicycle_rack': 9,
    }
    
    CLASSES = ('car', 'truck', 'bus', 'trailer', 'pedestrian', 
               'bicycle', 'motorcycle', 'traffic_cone', 'barrier', 'bicycle_rack')
    
    def __init__(
        self,
        data_root: str,
        ann_file: Optional[str] = None,
        pipeline: Optional[List] = None,
        test_mode: bool = False,
        modality: dict = None,
        box_type_3d: str = 'LiDAR',
        filter_empty_gt: bool = True,
        version: str = 'v1.0-mini',
    ):
        """
        Initialize nuScenes dataset.
        
        Args:
            data_root: Root directory of the dataset
            ann_file: Path to annotation file (JSON format, optional for nuScenes)
            pipeline: List of transforms to apply
            test_mode: Whether in test mode
            modality: Modality dict (e.g., {'use_lidar': True, 'use_camera': True})
            box_type_3d: Type of 3D box representation
            filter_empty_gt: Whether to filter samples with no ground truth
            version: nuScenes version (default: v1.0-mini)
        """
        if not NUSCENES_AVAILABLE:
            raise ImportError(
                "nuscenes-devkit is required. Install with: pip install nuscenes-devkit"
            )
        
        self.data_root = Path(data_root)
        self.version = version
        self.test_mode = test_mode
        self.modality = modality or {'use_lidar': True, 'use_camera': False}
        self.box_type_3d = box_type_3d
        self.filter_empty_gt = filter_empty_gt
        self.pipeline = pipeline or []
        
        # Initialize nuScenes
        logger.info(f"Loading nuScenes {version} from {data_root}...")
        self.nusc = NuScenes(version=version, dataroot=str(data_root), verbose=True)
        
        # Get scenes
        self.scenes = self.nusc.scene
        
        # Build sample index
        self.data_infos = self._build_sample_index()
        
        if filter_empty_gt and not test_mode:
            self.data_infos = [
                info for info in self.data_infos 
                if len(info.get('annos', {}).get('gt_boxes_3d', [])) > 0
            ]
            logger.info(f"Filtered to {len(self.data_infos)} samples with ground truth")
        
        logger.info(f"Loaded {len(self.data_infos)} samples from nuScenes {version}")
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all samples from nuScenes.
        
        Returns:
            List of sample info dictionaries
        """
        data_infos = []
        
        for scene_idx, scene in enumerate(self.scenes):
            # Get first sample token
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                
                # Get LiDAR data
                lidar_token = sample['data']['LIDAR_TOP']
                lidar_data = self.nusc.get('sample_data', lidar_token)
                
                # Get annotations
                annos = self._get_annotations(sample)
                
                info = {
                    'sample_token': sample_token,
                    'scene_token': scene['token'],
                    'lidar_token': lidar_token,
                    'lidar_path': lidar_data['filename'],
                    'timestamp': sample['timestamp'],
                    'annos': annos,
                }
                
                data_infos.append(info)
                
                # Get next sample
                sample_token = sample.get('next')
        
        return data_infos
    
    def _get_annotations(self, sample: Dict) -> Dict:
        """
        Get 3D bounding box annotations for a sample.
        
        Args:
            sample: nuScenes sample dictionary
            
        Returns:
            Annotation dictionary with boxes and labels
        """
        boxes_3d = []
        labels_3d = []
        
        # Get all annotations for this sample
        ann_tokens = sample['anns']
        
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get box
            box = self.nusc.get_box(ann_token)
            
            # Convert to our format: [x, y, z, l, w, h, yaw]
            center = box.center
            size = box.wlh  # width, length, height
            yaw = box.orientation.yaw_pitch_roll[0]  # yaw angle
            
            # Map nuScenes category to our class index
            category_name = ann['category_name']
            if category_name not in self.NUSCENES_TO_CLASS:
                # Try matching by prefix for unknown categories
                matched = False
                for nuscenes_cat, class_id in self.NUSCENES_TO_CLASS.items():
                    if category_name.startswith(nuscenes_cat.split('.')[0] + '.'):
                        class_id_to_use = class_id
                        matched = True
                        break
                if not matched:
                    continue  # Skip unknown classes
            else:
                class_id_to_use = self.NUSCENES_TO_CLASS[category_name]
            
            boxes_3d.append([
                center[0], center[1], center[2],
                size[0], size[1], size[2],
                yaw
            ])
            labels_3d.append(class_id_to_use)
        
        return {
            'gt_boxes_3d': boxes_3d,
            'gt_labels_3d': labels_3d,
        }
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a data sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - points: Point cloud [N, 4] (x, y, z, intensity)
                - gt_bboxes_3d: Ground truth 3D boxes
                - gt_labels_3d: Ground truth labels
                - sample_token: nuScenes sample token
        """
        info = self.data_infos[idx]
        
        # Load point cloud
        points = self._load_points(info['lidar_path'])
        
        # Load annotations
        annos = info.get('annos', {})
        if annos and not self.test_mode:
            gt_bboxes_3d = np.array(annos['gt_boxes_3d'], dtype=np.float32)
            gt_labels_3d = np.array(annos['gt_labels_3d'], dtype=np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
        
        # Prepare data dict
        data = {
            'points': points,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'sample_idx': idx,
            'sample_token': info['sample_token'],
            'pts_filename': info['lidar_path'],
        }
        
        # Apply pipeline transforms
        for transform in self.pipeline:
            data = transform(data)
        
        return data
    
    def _load_points(self, lidar_path: str) -> np.ndarray:
        """
        Load point cloud from nuScenes.
        
        Args:
            lidar_path: Path to point cloud file (relative to data_root)
            
        Returns:
            Point cloud array [N, 4] (x, y, z, intensity)
        """
        full_path = self.data_root / lidar_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {full_path}")
        
        # Load using nuscenes-devkit
        pc = LidarPointCloud.from_file(str(full_path))
        points = pc.points[:4].T  # [N, 4] with x, y, z, intensity
        
        return points.astype(np.float32)
    
    def get_data_info(self, idx: int) -> Dict:
        """Get raw data info by index."""
        return self.data_infos[idx]


def create_nuscenes_annotation_format(
    data_root: str,
    output_file: str,
    version: str = 'v1.0-mini'
):
    """
    Create MMDetection3D annotation file from nuScenes dataset.
    
    Args:
        data_root: Root directory of nuScenes dataset
        output_file: Output JSON file path
        version: nuScenes version
    """
    if not NUSCENES_AVAILABLE:
        logger.error("nuscenes-devkit not available")
        return
    
    logger.info(f"Converting nuScenes {version} to MMDetection3D format...")
    
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    
    annotation_list = []
    
    for scene in nusc.scene:
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # Get LiDAR data
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            
            # Get annotations
            boxes_3d = []
            labels_3d = []
            
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                box = nusc.get_box(ann_token)
                
                class_name = ann['category_name'].split('.')[0]
                
                boxes_3d.append({
                    'center': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.yaw_pitch_roll[0],
                    'class_id': class_name
                })
            
            annotation_list.append({
                'sample_token': sample_token,
                'lidar_path': lidar_data['filename'],
                'timestamp': sample['timestamp'],
                'gt_boxes_3d': boxes_3d,
            })
            
            sample_token = sample.get('next')
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotation_list, f, indent=2)
    
    logger.info(f"Saved annotations to {output_file}")
