"""3D point cloud visualization for predictions and ground truth."""

import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
import matplotlib.cm as cm
from pathlib import Path


class PointCloudVisualizer:
    """Visualize 3D point clouds with bounding boxes."""
    
    # Color mapping for nuScenes classes
    CLASS_COLORS = {
        'car': [1.0, 0.0, 0.0],  # Red
        'truck': [0.0, 1.0, 0.0],  # Green
        'bus': [0.0, 0.0, 1.0],  # Blue
        'trailer': [1.0, 1.0, 0.0],  # Yellow
        'construction_vehicle': [1.0, 0.0, 1.0],  # Magenta
        'pedestrian': [0.0, 1.0, 1.0],  # Cyan
        'motorcycle': [1.0, 0.5, 0.0],  # Orange
        'bicycle': [0.5, 0.0, 1.0],  # Purple
        'traffic_cone': [0.5, 0.5, 0.5],  # Gray
        'barrier': [0.0, 0.5, 0.5],  # Teal
    }
    
    def __init__(self, point_size: float = 2.0, background_color: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize visualizer.
        
        Args:
            point_size: Size of point cloud points
            background_color: Background color (R, G, B) in [0, 1]
        """
        self.point_size = point_size
        self.background_color = background_color
    
    def create_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud from numpy array.
        
        Args:
            points: (N, 3) array of xyz coordinates
            colors: (N, 3) array of RGB colors [0, 1] or None for intensity
            
        Returns:
            Open3D point cloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif points.shape[1] >= 4:
            # Use intensity as color (grayscale)
            intensity = points[:, 3] / points[:, 3].max() if points[:, 3].max() > 0 else points[:, 3]
            colors = np.stack([intensity, intensity, intensity], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def create_bbox(self, center: np.ndarray, size: np.ndarray, rotation: float, color: Tuple[float, float, float]) -> o3d.geometry.LineSet:
        """
        Create 3D bounding box wireframe.
        
        Args:
            center: (3,) center position [x, y, z]
            size: (3,) box dimensions [w, l, h]
            rotation: Rotation angle in radians (around z-axis)
            color: RGB color [0, 1]
            
        Returns:
            Open3D line set
        """
        # Create box corners in local coordinates
        w, l, h = size / 2.0
        corners = np.array([
            [-w, -l, -h], [w, -l, -h], [w, l, -h], [-w, l, -h],  # Bottom
            [-w, -l, h], [w, -l, h], [w, l, h], [-w, l, h]  # Top
        ])
        
        # Rotate around z-axis
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        corners = corners @ rotation_matrix.T
        
        # Translate to center
        corners = corners + center
        
        # Define edges (12 edges for a box)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(color)
        
        return line_set
    
    def visualize_scene(
        self,
        points: np.ndarray,
        gt_boxes: Optional[List[Dict]] = None,
        pred_boxes: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> o3d.visualization.Visualizer:
        """
        Visualize point cloud with ground truth and predictions.
        
        Args:
            points: (N, 3+) point cloud
            gt_boxes: List of ground truth boxes with keys: 'center', 'size', 'rotation', 'label'
            pred_boxes: List of prediction boxes with keys: 'center', 'size', 'rotation', 'label', 'score'
            save_path: Optional path to save image
            show: Whether to display interactively
            
        Returns:
            Open3D visualizer
        """
        # Create point cloud
        pcd = self.create_pointcloud(points)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800)
        vis.add_geometry(pcd)
        
        # Add ground truth boxes (green)
        if gt_boxes:
            for box in gt_boxes:
                color = self.CLASS_COLORS.get(box['label'], [1.0, 1.0, 1.0])
                bbox = self.create_bbox(
                    center=np.array(box['center']),
                    size=np.array(box['size']),
                    rotation=box['rotation'],
                    color=color
                )
                vis.add_geometry(bbox)
        
        # Add prediction boxes (red, with transparency effect via line width)
        if pred_boxes:
            for box in pred_boxes:
                color = self.CLASS_COLORS.get(box['label'], [1.0, 0.0, 0.0])
                # Make predictions slightly brighter
                color = [min(1.0, c * 1.2) for c in color]
                bbox = self.create_bbox(
                    center=np.array(box['center']),
                    size=np.array(box['size']),
                    rotation=box['rotation'],
                    color=color
                )
                vis.add_geometry(bbox)
        
        # Set background and point size
        opt = vis.get_render_option()
        opt.background_color = np.array(self.background_color)
        opt.point_size = self.point_size
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.7)
        
        if save_path:
            vis.capture_screen_image(str(save_path))
        
        if show:
            vis.run()
        else:
            vis.poll_events()
            vis.update_renderer()
        
        return vis
    
    @staticmethod
    def boxes_from_nuscenes_format(boxes: np.ndarray, labels: np.ndarray, scores: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Convert nuScenes format boxes to visualization format.
        
        Args:
            boxes: (N, 9) array [x, y, z, w, l, h, yaw, vx, vy] or (N, 7) [x, y, z, w, l, h, yaw]
            labels: (N,) array of class indices or names
            scores: Optional (N,) array of confidence scores
            
        Returns:
            List of box dictionaries
        """
        box_list = []
        class_names = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                       'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        
        for i in range(len(boxes)):
            box = boxes[i]
            label_idx = labels[i] if isinstance(labels[i], (int, np.integer)) else labels[i]
            label_name = class_names[label_idx] if isinstance(label_idx, (int, np.integer)) else label_idx
            
            box_dict = {
                'center': box[:3],
                'size': box[3:6],
                'rotation': box[6] if len(box) >= 7 else 0.0,
                'label': label_name
            }
            
            if scores is not None:
                box_dict['score'] = float(scores[i])
            
            box_list.append(box_dict)
        
        return box_list

