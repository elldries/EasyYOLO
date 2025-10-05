#!/usr/bin/env python3
"""
Train.py - YOLO Model Training Script
Supports YOLOv8 and YOLO11 models with versioning and interactive mode
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML not found. Please run DwAll.py to install dependencies.")
    sys.exit(1)

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("Error: ultralytics or torch not found. Please run DwAll.py to install dependencies.")
    sys.exit(1)


def check_and_install_cuda():
    """Check CUDA availability and provide installation guidance"""
    if not torch.cuda.is_available():
        print("\n⚠  CUDA is not available. Training will use CPU (slower).")
        print("\nTo enable CUDA support:")
        print("  1. Install NVIDIA GPU drivers: https://www.nvidia.com/Download/index.aspx")
        print("  2. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  3. Reinstall PyTorch with CUDA support:")
        print("     pip uninstall torch torchvision")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        response = input("\n► Continue with CPU training? ([Y]es/[N]o) [default: Y]: ").strip().lower()
        if response in ['n', 'no']:
            print("\n✓ Exiting. Install CUDA support and try again.\n")
            sys.exit(0)
        return False
    return True


class TrainingConfig:
    """Configuration for training session"""
    VALID_BASE_MODELS = [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
    ]
    
    def __init__(self):
        self.name = None
        self.version = None
        self.base_model = None
        self.epochs = None
        self.device = None
        self.resume = None
        self.data_folder = "DATA"
        self.model_registry_path = None
        self.model_output_dir = None


class ModelRegistry:
    """Manage model registry"""
    
    def __init__(self, registry_path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new one"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted registry file. Creating new one.")
                return {}
        return {}
    
    def save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def add_model(self, name, version, model_info):
        """Add or update model in registry"""
        key = f"{name}_{version}"
        self.models[key] = model_info
        self.save_registry()
    
    def get_model(self, name, version):
        """Get model info from registry"""
        key = f"{name}_{version}"
        return self.models.get(key)
    
    def list_models(self):
        """List all registered models"""
        return self.models


def scan_dataset_folders():
    """Scan DATA directory recursively for valid dataset folders, excluding paths containing 'models'"""
    data_dir = Path("DATA")
    
    if not data_dir.exists():
        data_dir = Path.cwd() / "DATA"
    
    if not data_dir.exists():
        return []
    
    valid_datasets = []
    excluded_folder_names = ['.git', '__pycache__', 'node_modules', '.venv', 'venv']
    
    def scan_recursive(current_dir):
        """Recursively scan directory for datasets"""
        try:
            for item in current_dir.iterdir():
                if not item.is_dir():
                    continue
                
                if item.name in excluded_folder_names:
                    continue
                
                if 'models' in str(item.relative_to(data_dir)).lower():
                    continue
                
                layout1_valid = all([
                    (item / "train" / "images").exists(),
                    (item / "train" / "labels").exists(),
                    (item / "val" / "images").exists(),
                    (item / "val" / "labels").exists()
                ])
                
                layout2_valid = all([
                    (item / "images" / "train").exists(),
                    (item / "labels" / "train").exists(),
                    (item / "images" / "val").exists(),
                    (item / "labels" / "val").exists()
                ])
                
                if layout1_valid or layout2_valid:
                    train_images = 0
                    if layout1_valid:
                        train_images = len(list((item / "train" / "images").glob("*")))
                    else:
                        train_images = len(list((item / "images" / "train").glob("*")))
                    
                    valid_datasets.append({
                        'path': str(item),
                        'name': item.name,
                        'images': train_images,
                        'layout': 'Layout 1' if layout1_valid else 'Layout 2'
                    })
                else:
                    scan_recursive(item)
        except (PermissionError, OSError):
            pass
    
    scan_recursive(data_dir)
    return valid_datasets


def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")


def validate_dataset(data_folder):
    """Validate dataset structure and YOLO label format"""
    print_header("Dataset Validation")
    
    data_path = Path(data_folder)
    
    def collect_images(image_dir):
        """Collect all image files from directory with common extensions"""
        images = []
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
        for ext in extensions:
            images.extend(list(image_dir.glob(f"*.{ext}")))
            images.extend(list(image_dir.glob(f"*.{ext.upper()}")))
        return images
    
    layout1_dirs = [
        data_path / "train" / "images",
        data_path / "train" / "labels",
        data_path / "val" / "images",
        data_path / "val" / "labels"
    ]
    
    layout2_dirs = [
        data_path / "images" / "train",
        data_path / "labels" / "train",
        data_path / "images" / "val",
        data_path / "labels" / "val"
    ]
    
    layout1_valid = all(d.exists() for d in layout1_dirs)
    layout2_valid = all(d.exists() for d in layout2_dirs)
    
    if layout1_valid:
        print("✓ Detected Layout 1: train/images, train/labels, val/images, val/labels")
        train_images = collect_images(data_path / "train" / "images")
        train_labels = list((data_path / "train" / "labels").glob("*.txt"))
        val_images = collect_images(data_path / "val" / "images")
        val_labels = list((data_path / "val" / "labels").glob("*.txt"))
    elif layout2_valid:
        print("✓ Detected Layout 2: images/train, labels/train, images/val, labels/val")
        train_images = collect_images(data_path / "images" / "train")
        train_labels = list((data_path / "labels" / "train").glob("*.txt"))
        val_images = collect_images(data_path / "images" / "val")
        val_labels = list((data_path / "labels" / "val").glob("*.txt"))
    else:
        print("⚠  Warning: Dataset does not match any supported YOLO layout")
        print("   Layout 1: train/images, train/labels, val/images, val/labels")
        print("   Layout 2: images/train, labels/train, images/val, labels/val")
        return False
    
    print(f"\n📊 Files Found:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Training labels: {len(train_labels)}")
    print(f"   Validation images: {len(val_images)}")
    print(f"   Validation labels: {len(val_labels)}")
    
    if not train_images or not train_labels:
        print("\n✗ Error: Training data is required but missing")
        if not train_images:
            print("   → No training images found (.jpg, .jpeg, .png, .bmp)")
        if not train_labels:
            print("   → No training labels found (.txt)")
        return False
    
    if not val_images or not val_labels:
        print("\n⚠  Warning: Validation data is missing or incomplete")
        if not val_images:
            print("   → No validation images found (.jpg, .jpeg, .png, .bmp)")
        if not val_labels:
            print("   → No validation labels found (.txt)")
        print("   Training will continue, but validation metrics may not be available")
    
    print(f"\n✓ Training images: {len(train_images)}")
    print(f"✓ Training labels: {len(train_labels)}")
    if val_images:
        print(f"✓ Validation images: {len(val_images)}")
    if val_labels:
        print(f"✓ Validation labels: {len(val_labels)}")
    
    sample_label = train_labels[0] if train_labels else None
    if sample_label:
        try:
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                            print(f"\n✓ Valid YOLO format detected")
        except Exception as e:
            print(f"⚠  Warning: Could not validate label format: {e}")
    
    print(f"{'='*70}\n")
    return True


def auto_split_dataset(data_folder):
    """Automatically split training data into train/val if validation is missing"""
    data_path = Path(data_folder)
    
    layout1_train_images = data_path / "train" / "images"
    layout1_train_labels = data_path / "train" / "labels"
    layout1_val_images = data_path / "val" / "images"
    layout1_val_labels = data_path / "val" / "labels"
    
    if layout1_train_images.exists() and layout1_train_labels.exists():
        if not layout1_val_images.exists() or not layout1_val_labels.exists():
            print("\n⚠  Validation data missing, creating 20% split from training data...")
            
            layout1_val_images.mkdir(parents=True, exist_ok=True)
            layout1_val_labels.mkdir(parents=True, exist_ok=True)
            
            import shutil
            import random
            
            all_images = list(layout1_train_images.glob("*"))
            random.shuffle(all_images)
            split_idx = int(len(all_images) * 0.8)
            
            val_images = all_images[split_idx:]
            
            for img_path in val_images:
                img_name = img_path.name
                label_name = img_path.stem + '.txt'
                label_path = layout1_train_labels / label_name
                
                if label_path.exists():
                    shutil.move(str(img_path), str(layout1_val_images / img_name))
                    shutil.move(str(label_path), str(layout1_val_labels / label_name))
            
            print(f"✓ Moved {len(val_images)} images to validation set")


def create_dataset_yaml(data_folder, num_classes=None):
    """Create dataset.yaml file"""
    data_path = Path(data_folder)
    yaml_path = data_path / "dataset.yaml"
    
    auto_split_dataset(data_folder)
    
    layout1_valid = (data_path / "train" / "labels").exists()
    layout2_valid = (data_path / "labels" / "train").exists()
    
    if num_classes is None:
        if layout1_valid:
            label_files = list((data_path / "train" / "labels").glob("*.txt"))
        else:
            label_files = list((data_path / "labels" / "train").glob("*.txt"))
        
        unique_classes = set()
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            unique_classes.add(class_id)
            except:
                pass
        
        num_classes = len(unique_classes)
        min_class = min(unique_classes) if unique_classes else 0
        max_class = max(unique_classes) if unique_classes else 0
    else:
        min_class = 0
        max_class = num_classes - 1
    
    class_names = {
        1: 'head',
        2: 'body'
    }
    
    names_list = []
    if min_class == 1 and max_class == 2:
        names_list = ['head', 'body']
    elif min_class == 0:
        names_list = [class_names.get(i, f'class_{i}') for i in range(num_classes)]
    else:
        names_list = [class_names.get(i+min_class, f'class_{i+min_class}') for i in range(num_classes)]
    
    if layout1_valid:
        dataset_config = {
            'path': str(data_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': num_classes,
            'names': names_list
        }
    else:
        dataset_config = {
            'path': str(data_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': num_classes,
            'names': names_list
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Created dataset.yaml ({num_classes} classes: {', '.join(names_list)})")
    
    return yaml_path


def train_model(config):
    """Train YOLO model"""
    print_header("Training")
    
    result = {
        "success": False,
        "model_path": None,
        "onnx_path": None,
        "epochs_completed": 0,
        "error": None
    }
    
    try:
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("⚠  Warning: CUDA requested but not available, falling back to CPU")
            config.device = 'cpu'
        
        if not validate_dataset(config.data_folder):
            result["error"] = "Dataset validation failed"
            print(f"⚠  {result['error']}\n")
            return result
        
        data_yaml = create_dataset_yaml(config.data_folder)
        
        if config.resume:
            print(f"\n► Resuming training from: {config.resume}")
            model = YOLO(config.resume)
        else:
            print(f"\n► Initializing model: {config.base_model}")
            model = YOLO(config.base_model)
        
        print(f"► Device: {config.device}")
        print(f"► Epochs: {config.epochs}")
        print(f"► Output: {config.model_output_dir}\n")
        
        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            device=config.device,
            project=str(config.model_output_dir.parent),
            name=config.model_output_dir.name,
            exist_ok=True,
            verbose=True
        )
        
        best_model_path = config.model_output_dir / "weights" / "best.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {best_model_path}")
        
        print(f"\n✓ Training completed successfully")
        print(f"✓ Model saved to: {best_model_path}")
        
        print(f"\n► Exporting to ONNX format...")
        model = YOLO(str(best_model_path))
        onnx_path = model.export(format='onnx')
        print(f"✓ ONNX model exported to: {onnx_path}")
        
        result["success"] = True
        result["model_path"] = str(best_model_path)
        result["onnx_path"] = str(onnx_path)
        result["epochs_completed"] = config.epochs
        
        registry = ModelRegistry(config.model_registry_path)
        model_info = {
            "name": config.name,
            "version": config.version,
            "base_model": config.base_model,
            "epochs": config.epochs,
            "device": config.device,
            "model_path": str(best_model_path),
            "onnx_path": str(onnx_path),
            "trained_at": datetime.now().isoformat(),
            "data_folder": config.data_folder
        }
        registry.add_model(config.name, config.version, model_info)
        print(f"✓ Model registered\n")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ Training failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    return result


def select_dataset():
    """Select dataset folder"""
    clear_screen()
    print_header("Dataset Selection")
    
    print("\n► Scanning for dataset folders...")
    datasets = scan_dataset_folders()
    
    if datasets:
        print(f"\nFound {len(datasets)} dataset folder(s):\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  [{i}] {ds['name']}")
            print(f"      Layout: {ds['layout']}, Images: {ds['images']}")
        
        print(f"\n  [{len(datasets) + 1}] Enter custom path")
        print(f"  [0] Use default (DATA)")
        
        while True:
            choice = input(f"\n► Select option [0-{len(datasets) + 1}]: ").strip()
            try:
                choice_idx = int(choice)
                if choice_idx == 0:
                    return "DATA"
                elif 1 <= choice_idx <= len(datasets):
                    selected = datasets[choice_idx - 1]['path']
                    print(f"✓ Selected: {selected}")
                    return selected
                elif choice_idx == len(datasets) + 1:
                    custom_path = input("\n► Enter dataset folder path: ").strip()
                    return custom_path if custom_path else "DATA"
                else:
                    print("✗ Invalid selection, try again")
            except ValueError:
                print("✗ Invalid input, try again")
    else:
        print("\n⚠  No dataset folders found in current directory")
        data_folder_input = input("► Enter dataset folder path [default: DATA]: ").strip()
        return data_folder_input if data_folder_input else "DATA"


def create_new_model(config, registry):
    """Create and train new model"""
    clear_screen()
    print_header("Create New Model")
    
    config.name = input("\n► Model name: ").strip()
    if not config.name:
        print("✗ Model name cannot be empty")
        return False
    
    config.version = input("► Version (e.g., v1.0): ").strip()
    if not config.version:
        print("✗ Version cannot be empty")
        return False
    
    print(f"\n{'─'*70}")
    print("Available Base Models:")
    print(f"{'─'*70}")
    for i, model in enumerate(TrainingConfig.VALID_BASE_MODELS, 1):
        size_desc = {
            'n': '⚡ Fastest, Least Accurate',
            's': '🚀 Fast, Good Accuracy',
            'm': '⚖️  Balanced Speed/Accuracy',
            'l': '🎯 Slow, High Accuracy',
            'x': '🔥 Slowest, Best Accuracy'
        }
        model_size = model.split('.')[0][-1]
        desc = size_desc.get(model_size, '')
        print(f"  [{i:2}] {model:<15} {desc}")
    
    while True:
        model_choice = input(f"\n► Select base model [1-{len(TrainingConfig.VALID_BASE_MODELS)}]: ").strip()
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(TrainingConfig.VALID_BASE_MODELS):
                config.base_model = TrainingConfig.VALID_BASE_MODELS[model_idx]
                break
            else:
                print("✗ Invalid selection, try again")
        except ValueError:
            print("✗ Invalid input, try again")
    
    while True:
        epochs_input = input("\n► Number of epochs [default: 100]: ").strip()
        try:
            config.epochs = int(epochs_input) if epochs_input else 100
            if config.epochs > 0:
                break
            print("✗ Epochs must be positive")
        except ValueError:
            print("✗ Invalid input, try again")
    
    cuda_available = check_and_install_cuda()
    if cuda_available:
        device_input = input("► Device ([1]=CUDA, [2]=CPU) [default: CUDA]: ").strip()
        config.device = 'cpu' if device_input == '2' else 'cuda'
    else:
        config.device = 'cpu'
    
    config.model_output_dir = Path(config.data_folder) / "models" / f"{config.name}_{config.version}"
    
    print(f"\n{'─'*70}")
    print("Training Configuration:")
    print(f"{'─'*70}")
    print(f"  Model Name:    {config.name}")
    print(f"  Version:       {config.version}")
    print(f"  Base Model:    {config.base_model}")
    print(f"  Epochs:        {config.epochs}")
    print(f"  Device:        {config.device}")
    print(f"  Output Dir:    {config.model_output_dir}")
    print(f"{'─'*70}")
    
    confirm = input("\n► Start training? ([Y]es/[N]o) [default: Y]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        result = train_model(config)
        print_header("Training Result")
        if result['success']:
            print(f"✓ Success!")
            print(f"✓ Model: {result['model_path']}")
            print(f"✓ ONNX: {result['onnx_path']}")
            print(f"✓ Epochs completed: {result['epochs_completed']}")
        else:
            print(f"✗ Failed: {result['error']}")
        print(f"{'='*70}")
        input("\n► Press Enter to continue...")
        return True
    return False


def continue_training_model(config, registry):
    """Continue training existing model"""
    clear_screen()
    print_header("Continue Training")
    
    models = registry.list_models()
    if not models:
        print("\n⚠  No registered models found")
        input("\n► Press Enter to continue...")
        return False
    
    print("\nRegistered Models:\n")
    model_list = list(models.items())
    for i, (key, info) in enumerate(model_list, 1):
        trained_date = info.get('trained_at', 'unknown')[:10]
        print(f"  [{i}] {info['name']} {info['version']}")
        print(f"      Base: {info['base_model']}, Trained: {trained_date}, Epochs: {info['epochs']}")
    
    while True:
        model_choice = input(f"\n► Select model [1-{len(model_list)}] or [0] to cancel: ").strip()
        try:
            model_idx = int(model_choice) - 1
            if model_choice == '0':
                return False
            if 0 <= model_idx < len(model_list):
                selected_key, selected_info = model_list[model_idx]
                break
            else:
                print("✗ Invalid selection, try again")
        except ValueError:
            print("✗ Invalid input, try again")
    
    config.name = selected_info['name']
    config.version = selected_info['version']
    config.base_model = selected_info['base_model']
    config.resume = selected_info['model_path']
    
    if not os.path.exists(config.resume):
        print(f"\n✗ Model file not found: {config.resume}")
        input("\n► Press Enter to continue...")
        return False
    
    while True:
        epochs_input = input("\n► Additional epochs [default: 50]: ").strip()
        try:
            config.epochs = int(epochs_input) if epochs_input else 50
            if config.epochs > 0:
                break
            print("✗ Epochs must be positive")
        except ValueError:
            print("✗ Invalid input, try again")
    
    cuda_available = check_and_install_cuda()
    if cuda_available:
        device_input = input("► Device ([1]=CUDA, [2]=CPU) [default: CUDA]: ").strip()
        config.device = 'cpu' if device_input == '2' else 'cuda'
    else:
        config.device = 'cpu'
    
    config.model_output_dir = Path(config.data_folder) / "models" / f"{config.name}_{config.version}"
    
    print(f"\n{'─'*70}")
    print("Continue Training Configuration:")
    print(f"{'─'*70}")
    print(f"  Model:         {config.name} {config.version}")
    print(f"  Resume from:   {config.resume}")
    print(f"  Add. Epochs:   {config.epochs}")
    print(f"  Device:        {config.device}")
    print(f"{'─'*70}")
    
    confirm = input("\n► Continue training? ([Y]es/[N]o) [default: Y]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        result = train_model(config)
        print_header("Training Result")
        if result['success']:
            print(f"✓ Success!")
            print(f"✓ Model: {result['model_path']}")
            print(f"✓ ONNX: {result['onnx_path']}")
            print(f"✓ Epochs completed: {result['epochs_completed']}")
        else:
            print(f"✗ Failed: {result['error']}")
        print(f"{'='*70}")
        input("\n► Press Enter to continue...")
        return True
    return False


def view_models(registry):
    """View all registered models"""
    clear_screen()
    print_header("Registered Models")
    
    models = registry.list_models()
    if not models:
        print("\n⚠  No registered models found")
    else:
        print()
        for i, (key, info) in enumerate(models.items(), 1):
            print(f"{'─'*70}")
            print(f"[{i}] {info['name']} {info['version']}")
            print(f"{'─'*70}")
            print(f"  Base Model:    {info['base_model']}")
            print(f"  Epochs:        {info['epochs']}")
            print(f"  Device:        {info['device']}")
            print(f"  Trained At:    {info.get('trained_at', 'unknown')[:19]}")
            print(f"  Model Path:    {info['model_path']}")
            print(f"  ONNX Path:     {info.get('onnx_path', 'N/A')}")
            print()
        print(f"{'='*70}")
    
    input("\n► Press Enter to continue...")


def interactive_mode():
    """Interactive mode for training"""
    config = TrainingConfig()
    
    config.data_folder = select_dataset()
    
    data_path = Path(config.data_folder)
    models_dir = data_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    config.model_registry_path = models_dir / "model_registry.json"
    
    registry = ModelRegistry(config.model_registry_path)
    
    while True:
        clear_screen()
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                     Lieris AI - YOLO Training                    ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        print(f"  Dataset: {config.data_folder}")
        print(f"  Registry: {len(registry.list_models())} model(s) registered\n")
        
        print(f"{'─'*70}")
        print("  [1]  Train New Model")
        print("  [2]  Continue Training Existing Model")
        print("  [3]  View Registered Models")
        print(f"{'─'*70}")
        print("  [0]  Exit")
        print(f"{'─'*70}")
        
        choice = input("\n► Select option [0-3]: ").strip()
        
        if choice == "1":
            create_new_model(config, registry)
        elif choice == "2":
            continue_training_model(config, registry)
        elif choice == "3":
            view_models(registry)
        elif choice == "0":
            print("\n✓ Exiting... Goodbye!\n")
            break
        else:
            print("\n✗ Invalid option, please try again")
            input("► Press Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='YOLO Model Training Script')
    parser.add_argument('--name', type=str, help='Model name')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--base-model', type=str, choices=TrainingConfig.VALID_BASE_MODELS,
                       help='Base model to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--data-folder', type=str, default='DATA', help='Data folder path')
    
    args = parser.parse_args()
    
    if args.name and args.version and args.base_model:
        config = TrainingConfig()
        config.name = args.name
        config.version = args.version
        config.base_model = args.base_model
        config.epochs = args.epochs
        
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("⚠  Warning: CUDA requested but not available, using CPU instead")
            config.device = 'cpu'
        else:
            config.device = args.device
        
        config.data_folder = args.data_folder
        
        data_path = Path(config.data_folder)
        models_dir = data_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        config.model_registry_path = models_dir / "model_registry.json"
        config.model_output_dir = models_dir / f"{config.name}_{config.version}"
        
        result = train_model(config)
        print(json.dumps(result))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
