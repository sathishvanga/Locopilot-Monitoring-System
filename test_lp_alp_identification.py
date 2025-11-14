"""
Test script for LP/ALP identification system

This script tests the new person role identification feature
that automatically identifies Loco Pilot (LP) and Assistant Loco Pilot (ALP)
based on detected objects near each person.
"""

import os
import sys
import json
from locopilot_monitor import LocopilotActivityMonitor

def test_lp_alp_identification():
    """Test LP/ALP identification on example video"""
    
    print("=" * 80)
    print("Testing LP/ALP Identification System")
    print("=" * 80)
    
    # Use an existing example video
    video_path = "example_data/latest.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Available videos in example_data/:")
        if os.path.exists("example_data"):
            for f in os.listdir("example_data"):
                if f.endswith(".mp4"):
                    print(f"  - {f}")
        return False
    
    print(f"\nProcessing video: {video_path}")
    print("-" * 80)
    
    # Create monitor with frame saving enabled to see role annotations
    monitor = LocopilotActivityMonitor(
        video_path,
        output_dir="locopilot_evidence",
        save_annotated_frames=True,  # Enable to see role annotations
        frame_save_interval=10,  # Save every 10th frame to see role labels
        sample_fps=0.5  # Sample at 0.5 FPS for faster processing
    )
    
    # Set trip/crew information
    monitor.trip_id = "TEST-LP-ALP-001"
    monitor.crew_name = "Test Crew"
    monitor.crew_id = "TEST-001"
    monitor.crew_role = 1
    
    # Process video
    print("\nStarting video processing...")
    print("Look for messages like: 'Person roles identified:'")
    print("-" * 80)
    
    monitor.process_video()
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    
    # Check results
    activities_file = os.path.join(monitor.run_dir, "activities.json")
    
    if os.path.exists(activities_file):
        with open(activities_file, 'r') as f:
            activities = json.load(f)
        
        print(f"\nTotal activities detected: {len(activities)}")
        
        # Check for person roles in activities
        activities_with_roles = [a for a in activities if 'personRoles' in a and a['personRoles']]
        
        print(f"Activities with person role information: {len(activities_with_roles)}")
        
        if activities_with_roles:
            print("\nSample activity with person roles:")
            print("-" * 80)
            sample_activity = activities_with_roles[0]
            print(f"Activity Type: {sample_activity['des']}")
            print(f"People Count: {sample_activity['peopleCount']}")
            
            if 'personRoles' in sample_activity and sample_activity['personRoles']:
                print("\nIdentified Roles:")
                for role_info in sample_activity['personRoles']:
                    print(f"  Person {role_info['personIndex'] + 1}: {role_info['roleName']}")
                    print(f"    - Role Code: {role_info['role']}")
                    print(f"    - LP Score: {role_info['lpScore']}")
                    print(f"    - ALP Score: {role_info['alpScore']}")
        else:
            print("\nNote: No activities with person role information found.")
            print("This may happen if:")
            print("  1. Only one person detected (default assigned as LP)")
            print("  2. No clear LP/ALP indicators present")
            print("  3. Detection confidence was low")
        
        print("\n" + "=" * 80)
        print("Results saved to:")
        print(f"  - Activities JSON: {activities_file}")
        print(f"  - Video clips: {monitor.evidence_clips_dir}")
        if monitor.save_annotated_frames:
            print(f"  - Annotated frames: {monitor.frames_dir}")
            print("\nCheck annotated frames to see person role labels (LP, ALP, etc.)")
        print("=" * 80)
        
        return True
    else:
        print(f"\nError: Activities file not found: {activities_file}")
        return False

def demonstrate_scoring_logic():
    """Demonstrate the LP/ALP scoring logic"""
    print("\n" + "=" * 80)
    print("LP/ALP Scoring Logic Explanation")
    print("=" * 80)
    
    print("""
The system identifies LP (Loco Pilot) and ALP (Assistant Loco Pilot) based on 
objects detected near each person:

LP Score Calculation (control-oriented objects):
  - TV/Monitor: +3 points (strong indicator of control station)
  - Laptop: +2 points
  - Keyboard: +2 points
  - Mouse: +1 point
  - Cell Phone: +1 point
  - Remote/Control Panel: +2 points

ALP Score Calculation (documentation-oriented objects):
  - Book/Logbook: +3 points (strong indicator of record-keeping)
  - Notebook: +3 points
  - Backpack: +1 point

Role Assignment:
  - If 1 person detected → Default to LP
  - If 2 people detected → Higher LP score = LP, other = ALP
  - If 3+ people detected → Highest LP score = LP, 2nd = ALP, others:
      * High ALP score → TRAINEE
      * High LP score (but not highest) → SUPERVISOR
      * Otherwise → VISITOR

Empty Desk Heuristic:
  - If no objects detected near a person, they get a slight ALP score boost
    (ALPs may have cleaner desks focused on documentation)
""")
    
    print("=" * 80)

if __name__ == "__main__":
    print("\n")
    demonstrate_scoring_logic()
    
    print("\n\nPress Enter to start testing with video data...")
    input()
    
    success = test_lp_alp_identification()
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test encountered issues. Please check the logs above.")
    
    print("\n")

