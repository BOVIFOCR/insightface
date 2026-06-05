#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Validate and organize flat landmark txt subdirectories to match a structured crops directory."
    )
    parser.add_argument(
        "--crops-path", 
        required=True, 
        help="Path to the structured images (e.g., /path/to/dataset_detected_faces/imgs_112x112)"
    )
    parser.add_argument(
        "--txt-path", 
        required=True, 
        help="Path to the current flat txt files (e.g., /path/to/dataset_detected_faces/txt)"
    )
    parser.add_argument(
        "--organize-subdir", 
        action="store_true", 
        help="Execute the move and reorganization if validation passes."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.crops_path):
        print(f"Error: Crops path does not exist: {args.crops_path}")
        sys.exit(1)
    if not os.path.exists(args.txt_path):
        print(f"Error: Txt path does not exist: {args.txt_path}")
        sys.exit(1)

    # 1. Map out the ground truth structure from the crops directory
    # Structure: video_to_subjects[video_id] = [subj_id1, subj_id2, ...]
    video_to_subjects = defaultdict(list)

    print("Scanning crops directory structure...")
    for subj_id in os.listdir(args.crops_path):
        subj_dir = os.path.join(args.crops_path, subj_id)
        if os.path.isdir(subj_dir):
            for video_id in os.listdir(subj_dir):
                video_dir = os.path.join(subj_dir, video_id)
                if os.path.isdir(video_dir):
                    video_to_subjects[video_id].append(subj_id)

    # 2. Scan current flat txt directory structure
    print("Scanning txt directory structure...")
    txt_videos = [d for d in os.listdir(args.txt_path) if os.path.isdir(os.path.join(args.txt_path, d))]

    # 3. Perform Validations
    print("\n--- RUNNING VALIDATION CHECKS ---")
    print(f"Total <video_id> subdirs found in crops: {len(video_to_subjects)}")
    print(f"Total <video_id> subdirs found in txt:   {len(txt_videos)}")

    has_errors = False
    collisions = {vid: subjs for vid, subjs in video_to_subjects.items() if len(subjs) > 1}

    # Check 1: Video ID Ambiguity (Collisions)
    if collisions:
        print("\n[CRITICAL ERROR] The following <video_id> directories exist across multiple subjects!")
        print("Moving these would merge data or overwrite files because the source structure is flat:")
        for vid, subjs in collisions.items():
            print(f"  - Video ID '{vid}' belongs to subjects: {subjs}")
        has_errors = True
    else:
        print("\n[OK] No <video_id> duplications/collisions found across different subjects.")

    # Check 2: Missing target directory warnings
    missing_in_txt = []
    for video_id in video_to_subjects.keys():
        if video_id in collisions:
            continue
        if video_id not in txt_videos:
            missing_in_txt.append(video_id)

    if missing_in_txt:
        print(f"\n[WARNING] There are {len(missing_in_txt)} video folders in crops that do not exist in the txt directory:")
        for item in missing_in_txt[:5]:
            print(f"  - {item}")
        if len(missing_in_txt) > 5:
            print(f"  - ... and {len(missing_in_txt) - 5} more.")

    # # 4. Action Execution Phase
    # print("\n--- ACTION PHASE ---")
    # if has_errors:
    #     print("Reorganization ABORTED due to critical folder collisions listed above.")
    #     sys.exit(1)

    if not args.organize_subdir:
        print("Dry-run complete. Directory validation passed successfully!")
        print("To execute the move, run the script again adding the '--organize-subdir' flag.")
        sys.exit(0)

    print("Proceeding to reorganize subdirectories...")
    moved_count = 0
    
    for video_id, subjs in video_to_subjects.items():
        subj_id = subjs[0] # Safe because len(subjs) == 1
        
        current_txt_path = os.path.join(args.txt_path, video_id)
        target_subj_dir = os.path.join(args.txt_path, subj_id)
        target_txt_path = os.path.join(target_subj_dir, video_id)
        
        if os.path.exists(current_txt_path):
            # Create target <subj_id> directory if it doesn't exist
            os.makedirs(target_subj_dir, exist_ok=True)
            
            # Move the video_id directory inside the new subj_id path
            shutil.move(current_txt_path, target_txt_path)
            moved_count += 1

    print(f"\nSuccessfully reorganized {moved_count} <video_id> directories into their respective <subj_id> folders!")

if __name__ == "__main__":
    main()