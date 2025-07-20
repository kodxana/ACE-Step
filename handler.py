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


# Global variables for model caching
MODEL_CACHE = {}
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")

# Check for preloaded model in container
PRELOADED_MODEL_PATH = "/workspace/models"
if os.path.exists(PRELOADED_MODEL_PATH):
    logger.info(f"Found preloaded model at: {PRELOADED_MODEL_PATH}")
    CHECKPOINT_PATH = PRELOADED_MODEL_PATH


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
            
            # Initialize pipeline
            pipeline = ACEStepPipeline(
                checkpoint_dir=checkpoint_path,
                dtype="bfloat16" if bf16 else "float32",
                torch_compile=torch_compile,
            )
            
            MODEL_CACHE[cache_key] = pipeline
            logger.info("Model loaded successfully and cached")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    return MODEL_CACHE[cache_key]


def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set default values for ACE-Step parameters.
    """
    # Required parameters with defaults
    defaults = {
        "audio_duration": 30.0,
        "prompt": "",
        "lyrics": "",
        "infer_step": 27,
        "guidance_scale": 3.0,
        "scheduler_type": "euler",
        "cfg_type": "double_condition",
        "omega_scale": 1.0,
        "actual_seeds": [42],
        "guidance_interval": 1.0,
        "guidance_interval_decay": 0.95,
        "min_guidance_scale": 1.0,
        "use_erg_tag": False,
        "use_erg_lyric": False,
        "use_erg_diffusion": False,
        "oss_steps": [5, 10, 15, 20, 25],
        "guidance_scale_text": 0.0,
        "guidance_scale_lyric": 0.0,
        # Model configuration
        "bf16": True,
        "torch_compile": False,
        "checkpoint_path": CHECKPOINT_PATH
    }
    
    # Apply defaults for missing values
    for key, default_value in defaults.items():
        if key not in job_input:
            job_input[key] = default_value
    
    # Validation
    if job_input["audio_duration"] <= 0 or job_input["audio_duration"] > 240:
        raise ValueError("audio_duration must be between 0 and 240 seconds")
    
    if job_input["infer_step"] < 1 or job_input["infer_step"] > 100:
        raise ValueError("infer_step must be between 1 and 100")
    
    if job_input["guidance_scale"] < 0 or job_input["guidance_scale"] > 10:
        raise ValueError("guidance_scale must be between 0 and 10")
    
    valid_schedulers = ["euler", "heun", "pingpong"]
    if job_input["scheduler_type"] not in valid_schedulers:
        raise ValueError(f"scheduler_type must be one of: {valid_schedulers}")
    
    valid_cfg_types = ["double_condition", "zero_star", "guidance"]
    if job_input["cfg_type"] not in valid_cfg_types:
        raise ValueError(f"cfg_type must be one of: {valid_cfg_types}")
    
    return job_input


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
        
        # Prepare generation parameters
        generation_params = (
            validated_input["audio_duration"],
            validated_input["prompt"],
            validated_input["lyrics"],
            validated_input["infer_step"],
            validated_input["guidance_scale"],
            validated_input["scheduler_type"],
            validated_input["cfg_type"],
            validated_input["omega_scale"],
            ", ".join(map(str, validated_input["actual_seeds"])),
            validated_input["guidance_interval"],
            validated_input["guidance_interval_decay"],
            validated_input["min_guidance_scale"],
            validated_input["use_erg_tag"],
            validated_input["use_erg_lyric"],
            validated_input["use_erg_diffusion"],
            ", ".join(map(str, validated_input["oss_steps"])),
            validated_input["guidance_scale_text"],
            validated_input["guidance_scale_lyric"],
        )
        
        # Generate audio with temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            logger.info("Starting audio generation...")
            model_pipeline(
                *generation_params,
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