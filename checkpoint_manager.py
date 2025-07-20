#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger
from typing import Optional


class CheckpointManager:
    """
    Manages model checkpoint downloads and caching for Runpod serverless workers.
    Handles both local storage and network volume storage.
    """
    
    def __init__(self, checkpoint_dir: str = "/runpod-volume/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Default Hugging Face model repository
        self.default_repo = "ACE-Step/ACE-Step-v1-3.5B"
        
    def get_checkpoint_path(self, model_name: Optional[str] = None) -> str:
        """
        Get the path to model checkpoints, downloading if necessary.
        
        Args:
            model_name: Optional specific model name/path. Uses default if None.
            
        Returns:
            Path to the model checkpoint directory
        """
        if model_name is None:
            model_name = self.default_repo
            
        # If it's a local path and exists, use it directly
        if os.path.exists(model_name):
            logger.info(f"Using local checkpoint path: {model_name}")
            return model_name
            
        # Check if model is already cached
        cached_path = self.checkpoint_dir / model_name.replace("/", "_")
        if cached_path.exists() and self._is_valid_checkpoint(cached_path):
            logger.info(f"Using cached checkpoint: {cached_path}")
            return str(cached_path)
            
        # Download model from Hugging Face
        return self._download_model(model_name, cached_path)
        
    def _download_model(self, repo_id: str, cache_path: Path) -> str:
        """
        Download model from Hugging Face Hub to cache directory.
        
        Args:
            repo_id: Hugging Face repository ID
            cache_path: Local cache path for the model
            
        Returns:
            Path to the downloaded model
        """
        logger.info(f"Downloading model {repo_id} to {cache_path}")
        
        try:
            # Download model using Hugging Face Hub
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_path.parent),
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Model downloaded successfully to {downloaded_path}")
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {str(e)}")
            raise e
            
    def _is_valid_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate that a checkpoint directory contains required files.
        
        Args:
            checkpoint_path: Path to check
            
        Returns:
            True if valid checkpoint, False otherwise
        """
        required_files = [
            "config.json",
            "pytorch_model.bin",  # or model.safetensors
        ]
        
        # Check for either .bin or .safetensors files
        has_model_file = (
            any((checkpoint_path / f).exists() for f in required_files) or
            any(checkpoint_path.glob("*.safetensors")) or
            any(checkpoint_path.glob("pytorch_model*.bin"))
        )
        
        has_config = (checkpoint_path / "config.json").exists()
        
        return has_model_file and has_config
        
    def cleanup_old_checkpoints(self, keep_latest: int = 2):
        """
        Clean up old cached checkpoints to save storage space.
        
        Args:
            keep_latest: Number of latest checkpoints to keep
        """
        if not self.checkpoint_dir.exists():
            return
            
        # Get all checkpoint directories sorted by modification time
        checkpoint_dirs = [
            d for d in self.checkpoint_dir.iterdir() 
            if d.is_dir() and self._is_valid_checkpoint(d)
        ]
        
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_dirs[keep_latest:]:
            logger.info(f"Removing old checkpoint: {old_checkpoint}")
            shutil.rmtree(old_checkpoint, ignore_errors=True)
            
    def get_storage_info(self) -> dict:
        """
        Get information about checkpoint storage usage.
        
        Returns:
            Dictionary with storage information
        """
        if not self.checkpoint_dir.exists():
            return {"total_size": 0, "checkpoint_count": 0, "checkpoints": []}
            
        checkpoints = []
        total_size = 0
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir() and self._is_valid_checkpoint(checkpoint_dir):
                size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
                checkpoints.append({
                    "name": checkpoint_dir.name,
                    "path": str(checkpoint_dir),
                    "size_mb": size / (1024 * 1024),
                    "modified": checkpoint_dir.stat().st_mtime
                })
                total_size += size
                
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoint_count": len(checkpoints),
            "checkpoints": sorted(checkpoints, key=lambda x: x["modified"], reverse=True)
        }


def preload_default_model():
    """
    Preload the default ACE-Step model for faster cold starts.
    This can be called during container initialization.
    """
    checkpoint_manager = CheckpointManager()
    try:
        checkpoint_path = checkpoint_manager.get_checkpoint_path()
        logger.info(f"Default model preloaded at: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logger.error(f"Failed to preload default model: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage and testing
    manager = CheckpointManager()
    
    # Get storage info
    storage_info = manager.get_storage_info()
    print(f"Storage info: {storage_info}")
    
    # Preload default model
    preload_default_model()