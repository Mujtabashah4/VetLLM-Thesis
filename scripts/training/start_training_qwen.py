#!/usr/bin/env python3
"""
Start QWEN training - tries full precision first, falls back to quantization if OOM
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def update_config_for_quantization(config_path):
    """Update config to enable quantization"""
    print("\n⚠️  Out of memory detected. Switching to quantization mode...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable quantization
    config['quantization']['enabled'] = True
    
    # Change optimizer to paged_adamw_8bit
    config['training']['optim'] = 'paged_adamw_8bit'
    
    # Increase batch size back (quantization uses less memory)
    config['training']['per_device_train_batch_size'] = 4
    config['training']['per_device_eval_batch_size'] = 4
    config['training']['gradient_accumulation_steps'] = 4
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Config updated for quantization mode")
    print("   - Quantization: enabled")
    print("   - Optimizer: paged_adamw_8bit")
    print("   - Batch size: 4 (increased)")

def main():
    """Main training function"""
    config_path = Path("experiments/qwen2.5-7b/configs/training_config.yaml")
    train_script = Path("experiments/shared/train.py")
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return 1
    
    print("="*80)
    print("QWEN FINE-TUNING - Starting with Full Precision")
    print("="*80)
    print("\nConfiguration:")
    print("  Mode: Full Precision (bfloat16)")
    print("  Quantization: Disabled")
    print("  Batch size: 2 per device")
    print("  Effective batch size: 16 (2 × 8 gradient accumulation)")
    print("\nIf out of memory occurs, will automatically switch to quantization mode.")
    print("="*80)
    
    # Run training
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path)
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        oom_detected = False
        for line in process.stdout:
            print(line, end='')
            
            # Check for OOM errors
            if any(keyword in line.lower() for keyword in [
                'out of memory',
                'cuda out of memory',
                'oom',
                'runtimeerror: cuda'
            ]):
                oom_detected = True
        
        process.wait()
        
        if process.returncode != 0 and oom_detected:
            print("\n" + "="*80)
            print("OUT OF MEMORY DETECTED")
            print("="*80)
            
            # Update config for quantization
            update_config_for_quantization(config_path)
            
            print("\n" + "="*80)
            print("RESTARTING WITH QUANTIZATION")
            print("="*80)
            
            # Restart training with quantization
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "="*80)
                print("✅ TRAINING COMPLETED WITH QUANTIZATION")
                print("="*80)
                return 0
            else:
                print("\n" + "="*80)
                print("❌ TRAINING FAILED")
                print("="*80)
                return 1
        
        elif process.returncode == 0:
            print("\n" + "="*80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY (FULL PRECISION)")
            print("="*80)
            return 0
        else:
            print("\n" + "="*80)
            print("❌ TRAINING FAILED")
            print("="*80)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

