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
            
            # Check for ACE-Step model structure
            required_subdirs = ['music_dcae_f8c8', 'music_vocoder', 'ace_step_transformer', 'umt5-base']
            missing_dirs = []
            
            for subdir in required_subdirs:
                subdir_path = model_dir / subdir
                if not subdir_path.exists():
                    missing_dirs.append(subdir)
                else:
                    logger.info(f"Found model component: {subdir}")
            
            if missing_dirs:
                logger.warning(f"⚠️ Missing model components: {missing_dirs}")
                return False
            else:
                logger.info("✅ Model validation passed - all required components found")
                return True
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