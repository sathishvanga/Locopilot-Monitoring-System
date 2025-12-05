"""
Benchmark Tier 4 optimizations

Compares performance across different optimization levels to measure
the impact of INT8 quantization and async frame reading.

Usage:
    python scripts/benchmark_tier4.py [--video VIDEO_PATH] [--iterations N]
"""

import time
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from locopilot_monitor import LocopilotActivityMonitor


def benchmark_config(video_path: str, config_name: str, env_vars: dict) -> dict:
    """
    Run benchmark with specific configuration.

    Args:
        video_path: Path to test video
        config_name: Name of configuration being tested
        env_vars: Environment variables to set

    Returns:
        Dictionary with benchmark results
    """
    # Set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    print("Configuration:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    print()

    try:
        # Run processing
        start_time = time.time()

        monitor = LocopilotActivityMonitor(video_path)
        activities = monitor.process_video()

        end_time = time.time()
        duration = end_time - start_time

        result = {
            'config': config_name,
            'duration': duration,
            'activities_count': len(activities) if activities else 0,
            'success': True
        }

        print(f"\n✅ Completed in {duration:.2f}s")
        print(f"   Activities detected: {result['activities_count']}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            'config': config_name,
            'duration': 0,
            'activities_count': 0,
            'success': False,
            'error': str(e)
        }

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def main():
    parser = argparse.ArgumentParser(description='Benchmark Tier 4 optimizations')
    parser.add_argument('--video', help='Path to test video (default: looks for videos in current dir)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations per config')
    args = parser.parse_args()

    # Find test video
    if args.video:
        video_path = args.video
    else:
        # Look for videos in common locations
        test_locations = [
            'sample_videos/*.mp4',
            'test_videos/*.mp4',
            '*.mp4'
        ]

        video_path = None
        for pattern in test_locations:
            videos = list(Path('.').glob(pattern))
            if videos:
                video_path = str(videos[0])
                break

    if not video_path or not os.path.exists(video_path):
        print("❌ No test video found!")
        print("   Specify a video with: --video path/to/video.mp4")
        print("   Or place a video in: sample_videos/, test_videos/, or current directory")
        return

    print("=" * 60)
    print("Tier 4 Optimizations Benchmark")
    print("=" * 60)
    print(f"Test video: {video_path}")
    print(f"Iterations: {args.iterations}")

    # Define benchmark configurations
    configs = {
        'Baseline (Tiers 1-3)': {
            'USE_ONNX_RUNTIME': '1',
            'USE_INT8_QUANTIZATION': '0',
            'USE_ASYNC_FRAME_READER': '0',
            'ENABLE_MOTION_SKIPPING': '1'
        },
        'Tier 4.1 (INT8)': {
            'USE_ONNX_RUNTIME': '1',
            'USE_INT8_QUANTIZATION': '1',
            'USE_ASYNC_FRAME_READER': '0',
            'ENABLE_MOTION_SKIPPING': '1'
        },
        'Tier 4.2 (Async I/O)': {
            'USE_ONNX_RUNTIME': '1',
            'USE_INT8_QUANTIZATION': '0',
            'USE_ASYNC_FRAME_READER': '1',
            'ENABLE_MOTION_SKIPPING': '1'
        },
        'All Optimizations (Tiers 1-4)': {
            'USE_ONNX_RUNTIME': '1',
            'USE_INT8_QUANTIZATION': '1',
            'USE_ASYNC_FRAME_READER': '1',
            'ENABLE_MOTION_SKIPPING': '1'
        }
    }

    # Run benchmarks
    all_results = []

    for config_name, env_vars in configs.items():
        config_results = []

        for iteration in range(args.iterations):
            if args.iterations > 1:
                print(f"\nIteration {iteration + 1}/{args.iterations}")

            result = benchmark_config(video_path, config_name, env_vars)
            config_results.append(result)

        # Calculate average if multiple iterations
        if len(config_results) > 0 and all(r['success'] for r in config_results):
            avg_duration = sum(r['duration'] for r in config_results) / len(config_results)
            avg_activities = sum(r['activities_count'] for r in config_results) / len(config_results)

            all_results.append({
                'config': config_name,
                'duration': avg_duration,
                'activities_count': avg_activities,
                'success': True
            })
        elif len(config_results) > 0:
            # At least one failed
            all_results.append({
                'config': config_name,
                'duration': 0,
                'activities_count': 0,
                'success': False
            })

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    if not all_results:
        print("No results to display")
        return

    baseline_duration = all_results[0]['duration'] if all_results[0]['success'] else None

    for result in all_results:
        print(f"\n{result['config']}:")

        if not result['success']:
            print("  ❌ Failed to complete")
            continue

        print(f"  Duration:   {result['duration']:.2f}s")

        if baseline_duration and baseline_duration > 0:
            speedup = baseline_duration / result['duration']
            improvement = ((baseline_duration - result['duration']) / baseline_duration) * 100
            print(f"  Speedup:    {speedup:.2f}x ({improvement:+.1f}%)")
        else:
            print(f"  Speedup:    N/A (baseline)")

        print(f"  Activities: {result['activities_count']:.0f}")

    # Print recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    if len(all_results) >= 4 and all(r['success'] for r in all_results):
        final_result = all_results[-1]  # All optimizations
        baseline_result = all_results[0]  # Baseline

        if baseline_result['duration'] > 0:
            total_speedup = baseline_result['duration'] / final_result['duration']

            if total_speedup >= 1.4:  # At least 40% faster
                print(f"✅ Tier 4 optimizations provide {total_speedup:.2f}x speedup!")
                print(f"   Enable in production: USE_INT8_QUANTIZATION=1, USE_ASYNC_FRAME_READER=1")
            else:
                print(f"⚠️  Limited benefit from Tier 4 ({total_speedup:.2f}x speedup)")
                print(f"   Current optimizations (Tiers 1-3) may be sufficient")

            # Check individual tiers
            if len(all_results) >= 2:
                int8_result = all_results[1]  # Tier 4.1
                if int8_result['success']:
                    int8_speedup = baseline_result['duration'] / int8_result['duration']
                    if int8_speedup >= 1.5:
                        print(f"\n💡 INT8 quantization very effective ({int8_speedup:.2f}x)")
                        print(f"   Consider enabling: USE_INT8_QUANTIZATION=1")

            if len(all_results) >= 3:
                async_result = all_results[2]  # Tier 4.2
                if async_result['success']:
                    async_speedup = baseline_result['duration'] / async_result['duration']
                    if async_speedup >= 1.2:
                        print(f"\n💡 Async I/O provides {async_speedup:.2f}x speedup")
                        print(f"   Consider enabling: USE_ASYNC_FRAME_READER=1")

        # Verify accuracy consistency
        activity_counts = [r['activities_count'] for r in all_results if r['success']]
        if activity_counts:
            max_count = max(activity_counts)
            min_count = min(activity_counts)
            if max_count > 0:
                variation = (max_count - min_count) / max_count * 100
                if variation <= 5:
                    print(f"\n✅ Activity detection consistent across configurations ({variation:.1f}% variation)")
                else:
                    print(f"\n⚠️  Activity detection varies significantly ({variation:.1f}% variation)")
                    print(f"   Review INT8 quantization accuracy")

    print()


if __name__ == "__main__":
    main()
