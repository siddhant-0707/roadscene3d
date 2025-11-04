#!/usr/bin/env python3
"""Evaluate trained PointPillars model and generate comprehensive report."""

import argparse
import json
import time
import torch
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log


def evaluate_model(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "work_dirs/evaluation",
    split: str = "val"
):
    """Run evaluation on trained model."""
    
    print("="*70)
    print("🔍 Model Evaluation - PointPillars Baseline")
    print("="*70)
    
    # Load config
    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    
    # Update output directory
    cfg.work_dir = output_dir
    
    # Switch to test mode
    if split == "test":
        cfg.test_dataloader.dataset.test_mode = True
        cfg.test_evaluator.out_dir = f"{output_dir}/test_results"
    else:
        cfg.val_dataloader.dataset.test_mode = True
        cfg.val_evaluator.out_dir = f"{output_dir}/val_results"
    
    # Create runner
    runner = Runner.from_cfg(cfg)
    
    # Run evaluation
    print(f"\n📊 Running evaluation on {split} set...")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📂 Output: {output_dir}")
    print()
    
    start_time = time.time()
    
    if split == "test":
        metrics = runner.test()
    else:
        metrics = runner.val()
    
    eval_time = time.time() - start_time
    
    print()
    print("="*70)
    print("✅ Evaluation Complete!")
    print(f"⏱️  Time: {eval_time:.1f}s")
    print("="*70)
    
    return metrics


def benchmark_inference_speed(config_path: str, checkpoint_path: str, num_samples: int = 100):
    """Benchmark inference speed (latency and throughput)."""
    
    print("\n" + "="*70)
    print("⚡ Inference Speed Benchmark")
    print("="*70)
    
    from mmengine.config import Config
    from mmengine.runner import Runner
    import torch
    
    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint_path
    
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint_path)
    model = runner.model
    
    # Get dataloader
    dataloader = runner.build_dataloader(cfg.val_dataloader)
    
    model.eval()
    
    print(f"\n🔬 Benchmarking on {num_samples} samples...")
    print("Warming up GPU...")
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        with torch.no_grad():
            _ = model.test_step(batch)
    
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        torch.cuda.synchronize()
        batch_start = time.time()
        
        with torch.no_grad():
            _ = model.test_step(batch)
        
        torch.cuda.synchronize()
        batch_time = time.time() - batch_start
        latencies.append(batch_time)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")
    
    total_time = time.time() - start_time
    
    latencies_ms = [l * 1000 for l in latencies]
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    min_latency = min(latencies_ms)
    max_latency = max(latencies_ms)
    p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    p99_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
    
    throughput = num_samples / total_time
    
    print("\n" + "="*70)
    print("📊 Inference Speed Results")
    print("="*70)
    print(f"  Average Latency:  {avg_latency:.2f} ms")
    print(f"  Min Latency:     {min_latency:.2f} ms")
    print(f"  Max Latency:     {max_latency:.2f} ms")
    print(f"  P95 Latency:     {p95_latency:.2f} ms")
    print(f"  P99 Latency:     {p99_latency:.2f} ms")
    print(f"  Throughput:      {throughput:.2f} samples/s ({throughput*60:.1f} samples/min)")
    print(f"  FPS equivalent:  {throughput:.1f} FPS")
    print("="*70)
    
    return {
        "avg_latency_ms": avg_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_samples_per_s": throughput,
        "fps": throughput
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointPillars model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pointpillars_nuscenes_mini.py",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dirs/pointpillars_nuscenes_baseline/epoch_40.pth",
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="work_dirs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference speed benchmark"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for benchmark"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_model(
        args.config,
        args.checkpoint,
        args.output_dir,
        args.split
    )
    
    # Run benchmark if requested
    speed_metrics = None
    if args.benchmark:
        speed_metrics = benchmark_inference_speed(
            args.config,
            args.checkpoint,
            args.num_samples
        )
    
    # Save results
    results = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "metrics": metrics,
    }
    
    if speed_metrics:
        results["speed_metrics"] = speed_metrics
    
    output_file = Path(args.output_dir) / "evaluation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()




