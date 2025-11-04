# nuScenes Mini Download Guide

## Required Downloads

### 1. nuScenes Mini Dataset (REQUIRED)
- **Path on website**: Full dataset (v1.0) → Mini
- **File**: "Metadata and sensor file blobs"
- **Size**: 3.88 GB
- **Download**: Click the download link
- **After download**: Extract to `data/nuscenes/`

```bash
mkdir -p data/nuscenes
cd data/nuscenes
# Extract the downloaded file
tar -xf ~/Downloads/v1.0-mini.tgz
# Or if it's a different format:
unzip ~/Downloads/v1.0-mini.zip
```

### 2. Map Expansion Pack (OPTIONAL but Recommended)
- **Path on website**: Map expansion
- **File**: "Map expansion pack (v1.3)"
- **Size**: 0.37 GB
- **Download**: Click download link
- **After download**: Extract to `data/nuscenes/maps/`

```bash
cd data/nuscenes
mkdir -p maps
cd maps
unzip ~/Downloads/maps.zip
```

## Expected Directory Structure

After extraction, you should have:

```
data/nuscenes/
├── v1.0-mini/
│   ├── maps/          # (if map expansion downloaded)
│   ├── samples/       # Camera and LiDAR keyframes
│   ├── sweeps/        # Non-keyframe data
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── sample.json
│   ├── scene.json
│   └── sensor.json
└── maps/              # Map expansion (optional)
```

## Verification

After downloading, verify the dataset:

```bash
source .venv/bin/activate

python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=True)
print(f'✅ Dataset loaded successfully!')
print(f'   Scenes: {len(nusc.scene)}')
print(f'   Samples: {len(nusc.sample)}')
print(f'   Annotations: {len(nusc.sample_annotation)}')
"
```

Expected output:
- Scenes: 10
- Samples: ~4,000
- Annotations: Many thousands

## Total Download Size

- **Minimum**: 3.88 GB (just mini dataset)
- **Recommended**: ~4.25 GB (mini + maps)

Much smaller than Waymo's 70-140 GB! 🎉

## Next Steps

After download and verification:

1. ✅ Dataset is ready
2. Update data loader configuration for nuScenes
3. Start training setup
4. Begin pipeline development

See [QUICKSTART.md](QUICKSTART.md) for next steps.
