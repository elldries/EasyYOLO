#!/usr/bin/env python3
"""
Folder.py - Dataset Folder Creator
Creates the necessary folder structure for YOLO training datasets
"""

import os
import sys
from pathlib import Path


def create_dataset_structure(folder_name, base_path="DATA"):
    """
    Create YOLO dataset folder structure
    
    Args:
        folder_name: Name of the dataset folder
        base_path: Base path where folders will be created (default: DATA)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dataset_path = Path(base_path) / folder_name
        
        if dataset_path.exists():
            print(f"Warning: Folder '{folder_name}' already exists in {base_path}/")
            response = input("Do you want to continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Operation cancelled.")
                return False
        
        directories = [
            dataset_path / "train" / "images",
            dataset_path / "train" / "labels",
            dataset_path / "val" / "images",
            dataset_path / "val" / "labels"
        ]
        
        print(f"\nCreating dataset structure for '{folder_name}'...")
        print(f"Location: {dataset_path.absolute()}")
        print("\nCreating directories:")
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            relative_path = directory.relative_to(base_path)
            print(f"  Created: {relative_path}")
        
        readme_path = dataset_path / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"Dataset: {folder_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write("This folder contains training data for YOLO AI model training.\n\n")
            
            f.write("FOLDER STRUCTURE:\n")
            f.write("=" * 80 + "\n")
            f.write("  train/images/ - Training images (screenshots from your game)\n")
            f.write("  train/labels/ - Training labels (YOLO format .txt files)\n")
            f.write("  val/images/   - Validation images (20-30% of total images)\n")
            f.write("  val/labels/   - Validation labels (YOLO format .txt files)\n\n")
            
            f.write("WHAT TO PUT IN IMAGES FOLDERS:\n")
            f.write("=" * 80 + "\n")
            f.write("  1. Images must show the targets you want to detect (enemies, heads, etc.)\n")
            f.write("  2. Recommended: 500-2000 images for good accuracy\n")
            f.write("  3. Image formats: .jpg, .png, .bmp\n")
            f.write("  4. Split: 70-80% in train/images, 20-30% in val/images\n\n")
            
            f.write("WHAT TO PUT IN LABELS FOLDERS:\n")
            f.write("=" * 80 + "\n")
            f.write("  1. Each image needs a corresponding .txt file with the SAME name\n")
            f.write("     Example: image001.jpg -> image001.txt\n")
            f.write("  2. Use labeling tools like:\n")
            f.write("     - LabelImg: https://github.com/tzutalin/labelImg\n")
            f.write("     - Roboflow: https://roboflow.com\n")
            f.write("     - CVAT: https://cvat.org\n\n")
            
            f.write("IMPORTANT NOTES:\n")
            f.write("=" * 80 + "\n")
            f.write("  - Each image MUST have a corresponding label file\n")
            f.write("  - Label file name must match image name (only extension differs)\n")
            f.write("  - Empty label files are OK (if image has no targets)\n")
            f.write("  - More training data = better accuracy\n")
            f.write("  - GPU training is MUCH faster than CPU\n\n")
        
        print(f"\n  Created: README.txt")
        
        print(f"\n{'='*60}")
        print("Dataset folder structure created successfully!")
        print(f"{'='*60}")
        print(f"\nLocation: {dataset_path.absolute()}")
        print("\nNext steps:")
        print("  1. Add training images to: train/images/")
        print("  2. Add training labels to: train/labels/")
        print("  3. Add validation images to: val/images/")
        print("  4. Add validation labels to: val/labels/")
        print("\nThen run Train.py to start training your model.")
        
        return True
        
    except Exception as e:
        print(f"\nError creating folder structure: {e}")
        return False


def interactive_mode():
    """Interactive mode for creating dataset folders"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   Dataset Folder Creator - Lieris                  ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("This tool creates the necessary folder structure for YOLO AI training.")
    print("All folders will be created in the DATA/ directory.")
    print("\nYou will need:")
    print("  • 500-2000 game/any screenshots showing targets (enemies, heads, etc.)")
    print("  • Labeling tool (LabelImg, Roboflow, or CVAT)")
    print("  • Label files in YOLO format (.txt files)\n")
    
    while True:
        folder_name = input("Enter dataset folder name (or 'exit' to quit): ").strip()
        
        if folder_name.lower() == 'exit':
            print("\nExiting...")
            return 0
        
        if not folder_name:
            print("Error: Folder name cannot be empty.\n")
            continue
        
        if any(char in folder_name for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            print("Error: Folder name contains invalid characters.\n")
            continue
        
        print(f"\nYou are about to create a dataset folder named: {folder_name}")
        print("This will create the following structure:")
        print(f"  DATA/{folder_name}/")
        print(f"    ├── train/")
        print(f"    │   ├── images/")
        print(f"    │   └── labels/")
        print(f"    └── val/")
        print(f"        ├── images/")
        print(f"        └── labels/")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        
        if confirm == 'y':
            success = create_dataset_structure(folder_name)
            if success:
                create_another = input("\nCreate another dataset folder? (y/n): ").strip().lower()
                if create_another != 'y':
                    print("\nExiting...")
                    return 0
            else:
                return 1
        else:
            print("\nOperation cancelled.\n")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        
        if folder_name in ['-h', '--help']:
            print("""
Dataset Folder Creator

Usage:
  python Folder.py [folder_name]
  python Folder.py             (interactive mode)

Arguments:
  folder_name    Name of the dataset folder to create

Examples:
  python Folder.py my_dataset
  python Folder.py             (will prompt for folder name)

This creates the following structure in DATA/:
  folder_name/
    ├── train/
    │   ├── images/    (place training images here)
    │   └── labels/    (place training labels here)
    └── val/
        ├── images/    (place validation images here)
        └── labels/    (place validation labels here)
            """)
            return 0
        
        if any(char in folder_name for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            print("Error: Folder name contains invalid characters.")
            print("Please use only alphanumeric characters, underscores, and hyphens.")
            return 1
        
        success = create_dataset_structure(folder_name)
        return 0 if success else 1
    else:
        return interactive_mode()


if __name__ == "__main__":
    sys.exit(main())
