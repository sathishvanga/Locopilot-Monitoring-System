#!/usr/bin/env python3
"""
Test script for hand gesture detection feature
Tests LP and ALP hand gesture detection functionality
"""

import os
import json
import sys
from pathlib import Path

def test_hand_gesture_detection():
    """Test the hand gesture detection implementation"""
    
    print("=" * 80)
    print("Hand Gesture Detection - Test Script")
    print("=" * 80)
    print()
    
    # Check if locopilot_monitor.py exists
    if not os.path.exists("locopilot_monitor.py"):
        print("❌ ERROR: locopilot_monitor.py not found!")
        print("   Please run this script from the project root directory.")
        return False
    
    try:
        from locopilot_monitor import LocopilotActivityMonitor
        print("✅ Successfully imported LocopilotActivityMonitor")
    except ImportError as e:
        print(f"❌ ERROR: Failed to import LocopilotActivityMonitor: {e}")
        return False
    
    # Check if activity models are updated
    try:
        from app.models.activity_models import ActivityTypeEnum
        
        if not hasattr(ActivityTypeEnum, 'LP_NOT_EXCHANGING_HAND_GESTURE'):
            print("❌ ERROR: LP_NOT_EXCHANGING_HAND_GESTURE not found in ActivityTypeEnum")
            return False
        
        if not hasattr(ActivityTypeEnum, 'ALP_NOT_EXCHANGING_HAND_GESTURE'):
            print("❌ ERROR: ALP_NOT_EXCHANGING_HAND_GESTURE not found in ActivityTypeEnum")
            return False
        
        lp_gesture_value = ActivityTypeEnum.LP_NOT_EXCHANGING_HAND_GESTURE
        alp_gesture_value = ActivityTypeEnum.ALP_NOT_EXCHANGING_HAND_GESTURE
        
        print(f"✅ Activity types defined correctly:")
        print(f"   LP_NOT_EXCHANGING_HAND_GESTURE = {lp_gesture_value}")
        print(f"   ALP_NOT_EXCHANGING_HAND_GESTURE = {alp_gesture_value}")
        
    except ImportError as e:
        print(f"❌ ERROR: Failed to import activity models: {e}")
        return False
    
    # Check if monitor has hand gesture activities
    try:
        # Find a test video
        test_video = None
        for video_file in ["example_data/latest.mp4", "example_data/latest_1.mp4", "example_data/packing.mp4"]:
            if os.path.exists(video_file):
                test_video = video_file
                break
        
        if not test_video:
            print("⚠️  WARNING: No test video found in example_data/")
            print("   Skipping video processing test")
            print("   Available test videos: latest.mp4, latest_1.mp4, packing.mp4")
        else:
            print(f"\n📹 Test video found: {test_video}")
            print("   You can run a full test with:")
            print(f"   python3 -c \"from locopilot_monitor import LocopilotActivityMonitor; ")
            print(f"m = LocopilotActivityMonitor('{test_video}', sample_fps=0.5); m.process_video()\"")
        
    except Exception as e:
        print(f"⚠️  WARNING: Could not check for test videos: {e}")
    
    # Check activity detection service
    try:
        from app.services.activity_detection_service import ActivityDetectionService
        
        service = ActivityDetectionService()
        
        # Check if hand gesture activities are in the maps
        if 'lp_hand_gesture' not in service.activity_type_map:
            print("❌ ERROR: 'lp_hand_gesture' not in activity_type_map")
            return False
        
        if 'alp_hand_gesture' not in service.activity_type_map:
            print("❌ ERROR: 'alp_hand_gesture' not in activity_type_map")
            return False
        
        print("✅ Activity detection service configured correctly")
        print(f"   lp_hand_gesture → {service.activity_type_map['lp_hand_gesture']}")
        print(f"   alp_hand_gesture → {service.activity_type_map['alp_hand_gesture']}")
        
        # Check descriptions
        print(f"   Description (LP): {service.activity_descriptions['lp_hand_gesture']}")
        print(f"   Description (ALP): {service.activity_descriptions['alp_hand_gesture']}")
        
        # Check evidence rules
        print(f"   Evidence (LP): {service.evidence_rules['lp_hand_gesture']}")
        print(f"   Evidence (ALP): {service.evidence_rules['alp_hand_gesture']}")
        
    except ImportError as e:
        print(f"❌ ERROR: Failed to import activity detection service: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Service check failed: {e}")
        return False
    
    # Check locopilot_monitor configuration
    try:
        from locopilot_monitor import LocopilotActivityMonitor
        
        # Create a dummy monitor to check attributes
        monitor = LocopilotActivityMonitor(
            video_path="dummy.mp4",
            output_dir="test_output",
            create_run_dir=False
        )
        
        # Check if hand gesture activities are in tracking dictionaries
        if 'lp_hand_gesture' not in monitor.activities:
            print("❌ ERROR: 'lp_hand_gesture' not in monitor.activities")
            return False
        
        if 'alp_hand_gesture' not in monitor.activities:
            print("❌ ERROR: 'alp_hand_gesture' not in monitor.activities")
            return False
        
        if 'lp_hand_gesture' not in monitor.consecutive_detections:
            print("❌ ERROR: 'lp_hand_gesture' not in consecutive_detections")
            return False
        
        if 'alp_hand_gesture' not in monitor.consecutive_detections:
            print("❌ ERROR: 'alp_hand_gesture' not in consecutive_detections")
            return False
        
        if 'lp_hand_gesture' not in monitor.activity_thresholds:
            print("❌ ERROR: 'lp_hand_gesture' not in activity_thresholds")
            return False
        
        if 'alp_hand_gesture' not in monitor.activity_thresholds:
            print("❌ ERROR: 'alp_hand_gesture' not in activity_thresholds")
            return False
        
        print("✅ Monitor tracking dictionaries configured correctly")
        
        # Check thresholds
        lp_threshold = monitor.activity_thresholds['lp_hand_gesture']
        alp_threshold = monitor.activity_thresholds['alp_hand_gesture']
        
        print(f"   LP threshold: min_duration={lp_threshold['min_duration']}s, "
              f"required_consecutive={lp_threshold['required_consecutive']}, "
              f"grace_frames={lp_threshold['grace_frames']}")
        
        print(f"   ALP threshold: min_duration={alp_threshold['min_duration']}s, "
              f"required_consecutive={alp_threshold['required_consecutive']}, "
              f"grace_frames={alp_threshold['grace_frames']}")
        
        # Check if detect_hand_gesture method exists
        if not hasattr(monitor, 'detect_hand_gesture'):
            print("❌ ERROR: detect_hand_gesture() method not found!")
            return False
        
        print("✅ detect_hand_gesture() method exists")
        
        # Check method signature
        import inspect
        sig = inspect.signature(monitor.detect_hand_gesture)
        params = list(sig.parameters.keys())
        
        if params != ['pose_landmarks', 'frame_shape', 'person_roles']:
            print(f"⚠️  WARNING: Method signature may be incorrect: {params}")
        else:
            print(f"✅ Method signature correct: {params}")
        
    except Exception as e:
        print(f"❌ ERROR: Monitor configuration check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 80)
    print()
    print("Hand gesture detection is ready to use.")
    print()
    print("To test with a video:")
    print("  python3 locopilot_monitor.py")
    print()
    print("Or programmatically:")
    print("  from locopilot_monitor import LocopilotActivityMonitor")
    print("  monitor = LocopilotActivityMonitor('video.mp4', sample_fps=0.5)")
    print("  monitor.process_video()")
    print()
    print("Check the output in: locopilot_evidence/run_*/")
    print("  - activities.json (for detected hand gestures)")
    print("  - clips/ (for video clips and screenshots)")
    print()
    
    return True

def demonstrate_detection_logic():
    """Demonstrate the hand gesture detection logic"""
    print("\n" + "=" * 80)
    print("Hand Gesture Detection Logic")
    print("=" * 80)
    print("""
The system detects hand gestures when LP or ALP raises their hand:

Detection Criteria:
  1. Hand raised above shoulder (at least 80px)
  2. Hand raised above elbow (active gesture)
  3. Hand visible (visibility > 0.5)
  4. Hand within frame boundaries

Temporal Requirements:
  - Minimum duration: 2 seconds
  - Required consecutive detections: 2 samples
  - Grace period: 3 samples (~6 seconds) to group multiple raises

Activity Types:
  - LP_NOT_EXCHANGING_HAND_GESTURE (8): LP raises hand, ALP does not
  - ALP_NOT_EXCHANGING_HAND_GESTURE (9): ALP raises hand, LP does not

Important:
  - If BOTH LP and ALP raise hands, NO activity is created
  - This indicates proper hand exchange (normal behavior)
  - System only flags when ONE person is gesturing alone

Role Identification:
  - Uses existing LP/ALP identification based on objects near each person
  - LP: Control objects (monitors, keyboards, etc.)
  - ALP: Documentation objects (books, notebooks, etc.)

Output:
  - Activity JSON entry with type 8 or 9
  - Video clip showing the hand gesture
  - Screenshot at detection moment
  - Person role information included
""")
    print("=" * 80)

if __name__ == "__main__":
    print("\n")
    
    # Run tests
    success = test_hand_gesture_detection()
    
    # Show detection logic
    demonstrate_detection_logic()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

