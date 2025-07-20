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
    
    # Apply defaults for missing values and ensure correct types
    for key, default_value in defaults.items():
        if key not in job_input:
            job_input[key] = default_value
    
    # Ensure string parameters are strings
    job_input["prompt"] = str(job_input["prompt"])
    job_input["lyrics"] = str(job_input["lyrics"])
    job_input["scheduler_type"] = str(job_input["scheduler_type"])
    job_input["cfg_type"] = str(job_input["cfg_type"])
    
    # Ensure numeric parameters are correct types
    job_input["audio_duration"] = float(job_input["audio_duration"])
    job_input["infer_step"] = int(job_input["infer_step"])
    job_input["guidance_scale"] = float(job_input["guidance_scale"])
    job_input["omega_scale"] = float(job_input["omega_scale"])
    job_input["guidance_interval"] = float(job_input["guidance_interval"])
    job_input["guidance_interval_decay"] = float(job_input["guidance_interval_decay"])
    job_input["min_guidance_scale"] = float(job_input["min_guidance_scale"])
    job_input["guidance_scale_text"] = float(job_input["guidance_scale_text"])
    job_input["guidance_scale_lyric"] = float(job_input["guidance_scale_lyric"])
    
    # Ensure list parameters are lists
    if not isinstance(job_input["actual_seeds"], list):
        job_input["actual_seeds"] = [job_input["actual_seeds"]]
    if not isinstance(job_input["oss_steps"], list):
        job_input["oss_steps"] = [job_input["oss_steps"]]
    
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
        
        # Prepare generation parameters - matching the pipeline __call__ signature
        # The pipeline expects named parameters, not positional args
        generation_kwargs = {
            "audio_duration": validated_input["audio_duration"],
            "prompt": validated_input["prompt"],
            "lyrics": validated_input["lyrics"],
            "infer_step": validated_input["infer_step"],
            "guidance_scale": validated_input["guidance_scale"],
            "scheduler_type": validated_input["scheduler_type"],
            "cfg_type": validated_input["cfg_type"],
            "omega_scale": validated_input["omega_scale"],
            "actual_seeds": ", ".join(map(str, validated_input["actual_seeds"])),
            "guidance_interval": validated_input["guidance_interval"],
            "guidance_interval_decay": validated_input["guidance_interval_decay"],
            "min_guidance_scale": validated_input["min_guidance_scale"],
            "use_erg_tag": validated_input["use_erg_tag"],
            "use_erg_lyric": validated_input["use_erg_lyric"],
            "use_erg_diffusion": validated_input["use_erg_diffusion"],
            "oss_steps": ", ".join(map(str, validated_input["oss_steps"])),
            "guidance_scale_text": validated_input["guidance_scale_text"],
            "guidance_scale_lyric": validated_input["guidance_scale_lyric"],
        }
        
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