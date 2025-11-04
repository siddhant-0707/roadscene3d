# Dataset Size Comparison

## Waymo Open Dataset

**Perception v1.4.3:**
- **Full dataset**: ~500+ GB
- **Subset (120 train + 25 val + 25 test)**: ~70-140 GB
- **Scenes**: ~2,030 segments (each 20 seconds)
- **Frames**: ~390,000 frames total

**Pros:**
- Industry-standard, production-grade data
- Excellent quality and annotations
- Very relevant for autonomous driving (Kodiak uses similar data)
- Large scale for training

**Cons:**
- Large download size even for subset
- Long download time
- High storage requirements

## nuScenes Mini

**nuScenes mini:**
- **Size**: ~1.5 GB
- **Scenes**: 10 scenes (subset of full dataset)
- **Frames**: ~4,000 frames

**Full nuScenes:**
- **Size**: ~400 GB (1,000 scenes)
- **Frames**: ~40,000 frames

**Pros:**
- Very small download size
- Quick to download and iterate
- Good for development and testing
- Well-documented and widely used
- Includes LiDAR, camera, and radar data

**Cons:**
- Less data for final training
- Smaller scale (but can scale up to full nuScenes later)
- Different annotation format (may need different conversion)

## Recommendation for Your Project

### Option 1: Start with nuScenes Mini (Recommended for Initial Development)
- ✅ Fast download (~1.5 GB vs 70-140 GB)
- ✅ Quick iteration cycles
- ✅ Perfect for prototyping and pipeline development
- ✅ Can demonstrate full pipeline quickly
- ⚠️ Less impressive for portfolio (but you can scale up)

### Option 2: Use Waymo Subset (For Final Demo)
- ✅ More impressive for portfolio
- ✅ Production-grade data
- ✅ Better aligned with Kodiak's work
- ⚠️ Larger download/storage requirements
- ⚠️ Slower iteration

### Option 3: Hybrid Approach (Best of Both)
1. **Development Phase**: Use nuScenes mini (~1.5 GB)
   - Fast iteration
   - Build complete pipeline
   - Test all components

2. **Final Demo**: Use Waymo subset (~70-140 GB)
   - Show production-scale capability
   - More impressive for portfolio
   - Better aligns with Kodiak

## Storage Requirements

| Dataset | Download Size | Storage Needed |
|---------|--------------|----------------|
| nuScenes mini | 1.5 GB | ~2-3 GB (with processing) |
| Waymo subset | 70-140 GB | ~100-150 GB (with processing) |
| Full Waymo | 500+ GB | ~600 GB+ |

## Download Time Estimates

| Dataset | Size | Time (50 Mbps) | Time (100 Mbps) |
|---------|------|----------------|----------------|
| nuScenes mini | 1.5 GB | ~4 minutes | ~2 minutes |
| Waymo subset | 100 GB | ~4.5 hours | ~2.2 hours |
| Full Waymo | 500 GB | ~22 hours | ~11 hours |

*Note: Actual times vary based on network conditions and GCP bucket performance*

## Next Steps

If you want to switch to nuScenes mini:

1. I can help you:
   - Download nuScenes mini dataset
   - Create data loader for nuScenes format
   - Adapt the pipeline for nuScenes

2. Or continue with Waymo:
   - Reduce subset size (e.g., 50 train + 10 val + 10 test = ~30-50 GB)
   - Download overnight/weekend
   - Use for final demo

Would you like to switch to nuScenes mini, or reduce the Waymo subset size?
