#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from loguru import logger

# Set up environment for model download
os.environ['CHECKPOINT_PATH'] = '/workspace/models'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def main():
    """Preload ACE-Step model during container build."""
    try:
        from checkpoint_manager import CheckpointManager
        
        # Create checkpoint manager
        manager = CheckpointManager('/workspace/models')
        
        # Download/cache the default model
        logger.info("Starting model preload...")
        model_path = manager.get_checkpoint_path()
        logger.info(f"✅ ACE-Step model preloaded successfully at: {model_path}")
        
        # Verify model files exist
        model_dir = Path(model_path)
        if model_dir.exists():
            files = list(model_dir.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
            logger.info(f"Model directory contains {file_count} files, total size: {total_size:.2f}GB")
            
            # Check for required files
            required_files = ['config.json']
            has_model_files = any(model_dir.glob('*.safetensors')) or any(model_dir.glob('pytorch_model*.bin'))
            has_config = (model_dir / 'config.json').exists()
            
            if has_model_files and has_config:
                logger.info("✅ Model validation passed - all required files found")
                return True
            else:
                logger.warning("⚠️ Model validation failed - missing required files")
                return False
        else:
            logger.error(f"❌ Model directory not found: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model preload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logger.warning("Model preload failed, but continuing build (model will download at runtime)")
        # Don't exit with error code to allow build to continue
    else:
        logger.info("Model preload completed successfully!")