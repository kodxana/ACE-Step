#!/usr/bin/env python3

import runpod
import os
import gc
import base64
import tempfile
import traceback
from typing import Dict, Any, Optional, List
from loguru import logger
import torch

from acestep.pipeline_ace_step import ACEStepPipeline
from rp_schema import validate_input as validate_schema, get_pipeline_kwargs


# Global variables for model caching
MODEL_CACHE = {}
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")

# Check for preloaded model in container
PRELOADED_MODEL_PATH = "/workspace/models"
if os.path.exists(PRELOADED_MODEL_PATH):
    # Verify it's a complete ACE-Step model
    required_components = ['music_dcae_f8c8', 'music_vocoder', 'ace_step_transformer', 'umt5-base']
    model_path = None
    
    # Check if it's directly the model directory
    if all(os.path.exists(os.path.join(PRELOADED_MODEL_PATH, comp)) for comp in required_components):
        model_path = PRELOADED_MODEL_PATH
        logger.info(f"Found complete preloaded model at: {PRELOADED_MODEL_PATH}")
    else:
        # Check if it's in a subdirectory (HuggingFace download structure)
        for item in os.listdir(PRELOADED_MODEL_PATH):
            item_path = os.path.join(PRELOADED_MODEL_PATH, item)
            if os.path.isdir(item_path):
                if all(os.path.exists(os.path.join(item_path, comp)) for comp in required_components):
                    model_path = item_path
                    logger.info(f"Found complete preloaded model at: {model_path}")
                    break
    
    if model_path:
        CHECKPOINT_PATH = model_path
    else:
        logger.warning(f"Preloaded model directory exists but incomplete, will download at runtime: {PRELOADED_MODEL_PATH}")
else:
    logger.info(f"No preloaded model found, will download at runtime to: {CHECKPOINT_PATH}")


def load_model(checkpoint_path: str, bf16: bool = True, torch_compile: bool = False) -> ACEStepPipeline:
    """
    Load and cache the ACE-Step model to minimize cold start times.
    """
    cache_key = f"{checkpoint_path}_{bf16}_{torch_compile}"
    
    if cache_key not in MODEL_CACHE:
        logger.info(f"Loading ACE-Step model from {checkpoint_path}")
        try:
            # Set environment for single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            # Initialize pipeline with explicit checkpoint path
            pipeline = ACEStepPipeline(
                checkpoint_dir=checkpoint_path,
                dtype="bfloat16" if bf16 else "float32",
                torch_compile=torch_compile,
            )
            
            # Ensure the pipeline uses the correct checkpoint path
            if hasattr(pipeline, 'checkpoint_dir'):
                pipeline.checkpoint_dir = checkpoint_path
            
            MODEL_CACHE[cache_key] = pipeline
            logger.info("Model loaded successfully and cached")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    return MODEL_CACHE[cache_key]


def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input using schema and add model configuration.
    """
    # Use schema validation
    validated_input = validate_schema(job_input)
    
    # Add model configuration parameters
    validated_input["bf16"] = validated_input.get("bf16", True)
    validated_input["torch_compile"] = validated_input.get("torch_compile", False)
    validated_input["checkpoint_path"] = job_input.get("checkpoint_path", CHECKPOINT_PATH)
    
    return validated_input


def cleanup_gpu_memory():
    """
    Clean up GPU memory between job executions.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def encode_audio_to_base64(file_path: str) -> str:
    """
    Encode audio file to base64 for Runpod output.
    """
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()
        return base64.b64encode(audio_data).decode('utf-8')


def handler(job):
    """
    Main handler function for Runpod serverless worker.
    Processes music generation requests using ACE-Step.
    """
    job_input = job['input']
    
    try:
        # Validate input parameters
        validated_input = validate_input(job_input)
        logger.info(f"Processing job with validated input: {validated_input}")
        
        # Load model (cached if available)
        model_pipeline = load_model(
            validated_input["checkpoint_path"],
            validated_input["bf16"],
            validated_input["torch_compile"]
        )
        
        # Get pipeline kwargs from validated input
        generation_kwargs = get_pipeline_kwargs(validated_input)
        
        # Generate audio with temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            logger.info("Starting audio generation...")
            logger.info(f"Generation kwargs: {generation_kwargs}")
            
            # Call pipeline with keyword arguments
            model_pipeline(
                **generation_kwargs,
                save_path=temp_path
            )
            logger.info("Audio generation completed")
            
            # Encode audio to base64 for output
            audio_base64 = encode_audio_to_base64(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
        
        # Clean up GPU memory
        cleanup_gpu_memory()
        
        return {
            "status": "success",
            "audio_base64": audio_base64,
            "parameters_used": validated_input,
            "message": "Audio generated successfully"
        }
        
    except Exception as e:
        error_msg = f"Error processing job: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        # Clean up GPU memory even on error
        cleanup_gpu_memory()
        
        return {
            "status": "error",
            "error": error_msg,
            "message": "Failed to generate audio"
        }


if __name__ == "__main__":
    logger.info("Starting ACE-Step Runpod Serverless Worker")
    runpod.serverless.start({"handler": handler})