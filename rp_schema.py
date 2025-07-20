"""
RunPod Input Schema for ACE-Step Serverless Worker

This schema defines all expected input parameters with their types,
defaults, and validation rules based on the ACEStepPipeline.__call__ method.
"""

INPUT_SCHEMA = {
    # Audio Generation Parameters
    "audio_duration": {
        "type": float,
        "required": False,
        "default": 30.0,
        "constraints": lambda x: 0 < x <= 240,
        "description": "Duration of generated audio in seconds (0-240)"
    },
    "prompt": {
        "type": str,
        "required": False,
        "default": "",
        "description": "Text description of the music style/genre"
    },
    "lyrics": {
        "type": str,
        "required": False,
        "default": "",
        "description": "Song lyrics with structure tags like [verse], [chorus]"
    },
    
    # Generation Control Parameters
    "infer_step": {
        "type": int,
        "required": False,
        "default": 27,
        "constraints": lambda x: 1 <= x <= 100,
        "description": "Number of inference steps (1-100)"
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 3.0,
        "constraints": lambda x: 0 <= x <= 10,
        "description": "Guidance strength (0-10)"
    },
    "scheduler_type": {
        "type": str,
        "required": False,
        "default": "euler",
        "choices": ["euler", "heun", "pingpong"],
        "description": "Scheduler algorithm"
    },
    "cfg_type": {
        "type": str,
        "required": False,
        "default": "double_condition",
        "choices": ["double_condition", "zero_star", "guidance", "apg", "cfg", "cfg_star"],
        "description": "Classifier-free guidance type"
    },
    "omega_scale": {
        "type": float,
        "required": False,
        "default": 1.0,
        "description": "Omega scaling factor"
    },
    
    # Seeds and Randomization
    "manual_seeds": {
        "type": list,
        "required": False,
        "default": [42],
        "description": "Random seeds for generation",
        "element_type": int
    },
    
    # Guidance Parameters
    "guidance_interval": {
        "type": float,
        "required": False,
        "default": 1.0,
        "description": "Guidance interval"
    },
    "guidance_interval_decay": {
        "type": float,
        "required": False,
        "default": 0.95,
        "description": "Guidance interval decay rate"
    },
    "min_guidance_scale": {
        "type": float,
        "required": False,
        "default": 1.0,
        "description": "Minimum guidance scale"
    },
    
    # ERG (Enhanced Representation Guidance) Features
    "use_erg_tag": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable ERG for tags"
    },
    "use_erg_lyric": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable ERG for lyrics"
    },
    "use_erg_diffusion": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable ERG for diffusion"
    },
    
    # OSS Steps Configuration
    "oss_steps": {
        "type": str,
        "required": False,
        "default": "",  # Empty string to avoid custom sigmas by default
        "description": "Comma-separated OSS step values (leave empty for standard timesteps)",
        "process": lambda x: x if isinstance(x, str) else ",".join(map(str, x))
    },
    
    # Additional Guidance Scales
    "guidance_scale_text": {
        "type": float,
        "required": False,
        "default": 0.0,
        "description": "Text-specific guidance scale"
    },
    "guidance_scale_lyric": {
        "type": float,
        "required": False,
        "default": 0.0,
        "description": "Lyric-specific guidance scale"
    },
    
    # Model Configuration
    "bf16": {
        "type": bool,
        "required": False,
        "default": True,
        "description": "Use bfloat16 precision"
    },
    "torch_compile": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable torch.compile optimization"
    },
    
    # Advanced Features (for future implementation)
    "format": {
        "type": str,
        "required": False,
        "default": "wav",
        "choices": ["wav", "mp3", "ogg"],
        "description": "Output audio format"
    },
    "batch_size": {
        "type": int,
        "required": False,
        "default": 1,
        "description": "Batch size for generation"
    },
    
    # Audio2Audio Parameters (optional)
    "audio2audio_enable": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Enable audio-to-audio mode"
    },
    "ref_audio_strength": {
        "type": float,
        "required": False,
        "default": 0.5,
        "constraints": lambda x: 0 <= x <= 1,
        "description": "Reference audio strength (0-1)"
    },
    "ref_audio_input": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Path or base64 encoded reference audio"
    },
    
    # LoRA Parameters (optional)
    "lora_name_or_path": {
        "type": str,
        "required": False,
        "default": "none",
        "description": "LoRA model name or path"
    },
    "lora_weight": {
        "type": float,
        "required": False,
        "default": 1.0,
        "constraints": lambda x: 0 <= x <= 2,
        "description": "LoRA weight (0-2)"
    },
    
    # Retake/Variation Parameters (optional)
    "retake_seeds": {
        "type": list,
        "required": False,
        "default": None,
        "description": "Seeds for retake variations",
        "element_type": int
    },
    "retake_variance": {
        "type": float,
        "required": False,
        "default": 0.5,
        "constraints": lambda x: 0 <= x <= 1,
        "description": "Variance for retake (0-1)"
    },
    
    # Task Type
    "task": {
        "type": str,
        "required": False,
        "default": "text2music",
        "choices": ["text2music", "audio2audio", "repaint", "edit", "extend"],
        "description": "Generation task type"
    },
    
    # Repaint Parameters (optional)
    "repaint_start": {
        "type": int,
        "required": False,
        "default": 0,
        "description": "Repaint start time in seconds"
    },
    "repaint_end": {
        "type": int,
        "required": False,
        "default": 0,
        "description": "Repaint end time in seconds"
    },
    "src_audio_path": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Source audio path for repaint/edit"
    },
    
    # Edit Parameters (optional)
    "edit_target_prompt": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Target prompt for editing"
    },
    "edit_target_lyrics": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Target lyrics for editing"
    },
    "edit_n_min": {
        "type": float,
        "required": False,
        "default": 0.0,
        "constraints": lambda x: 0 <= x <= 1,
        "description": "Minimum edit strength (0-1)"
    },
    "edit_n_max": {
        "type": float,
        "required": False,
        "default": 1.0,
        "constraints": lambda x: 0 <= x <= 1,
        "description": "Maximum edit strength (0-1)"
    },
    "edit_n_avg": {
        "type": int,
        "required": False,
        "default": 1,
        "description": "Average edit steps"
    }
}


def validate_input(job_input: dict) -> dict:
    """
    Validate and process input according to schema.
    
    Args:
        job_input: Raw input dictionary from RunPod
        
    Returns:
        Validated and processed input dictionary
        
    Raises:
        ValueError: If validation fails
    """
    validated = {}
    
    for key, schema in INPUT_SCHEMA.items():
        # Get value from input or use default
        if key in job_input:
            value = job_input[key]
        else:
            value = schema.get("default")
            if value is None and schema.get("required", False):
                raise ValueError(f"Required parameter '{key}' is missing")
        
        # Skip None values for optional parameters
        if value is None and not schema.get("required", False):
            continue
            
        # Type checking and conversion
        expected_type = schema["type"]
        if not isinstance(value, expected_type):
            try:
                if expected_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes')
                elif expected_type == list:
                    if isinstance(value, str):
                        value = [int(x.strip()) for x in value.split(',')]
                    elif not isinstance(value, list):
                        value = [value]
                else:
                    value = expected_type(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Parameter '{key}' must be of type {expected_type.__name__}: {e}")
        
        # Check element types for lists
        if expected_type == list and "element_type" in schema:
            element_type = schema["element_type"]
            try:
                value = [element_type(x) for x in value]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Elements of '{key}' must be of type {element_type.__name__}: {e}")
        
        # Apply constraints
        if "constraints" in schema:
            if not schema["constraints"](value):
                raise ValueError(f"Parameter '{key}' failed validation constraint")
        
        # Check choices
        if "choices" in schema and value not in schema["choices"]:
            raise ValueError(f"Parameter '{key}' must be one of: {schema['choices']}")
        
        # Apply processing function if exists
        if "process" in schema:
            value = schema["process"](value)
        
        validated[key] = value
    
    # Handle special parameter mappings for backwards compatibility
    if "actual_seeds" in job_input and "manual_seeds" not in validated:
        validated["manual_seeds"] = job_input["actual_seeds"]
        if not isinstance(validated["manual_seeds"], list):
            validated["manual_seeds"] = [validated["manual_seeds"]]
    
    return validated


def get_pipeline_kwargs(validated_input: dict) -> dict:
    """
    Convert validated input to kwargs for ACEStepPipeline.__call__
    
    Args:
        validated_input: Validated input dictionary
        
    Returns:
        Dictionary of kwargs for pipeline
    """
    # Map of input keys to pipeline parameter names
    pipeline_params = {
        "audio_duration": "audio_duration",
        "prompt": "prompt",
        "lyrics": "lyrics",
        "infer_step": "infer_step",
        "guidance_scale": "guidance_scale",
        "scheduler_type": "scheduler_type",
        "cfg_type": "cfg_type",
        "omega_scale": "omega_scale",
        "manual_seeds": "manual_seeds",
        "guidance_interval": "guidance_interval",
        "guidance_interval_decay": "guidance_interval_decay",
        "min_guidance_scale": "min_guidance_scale",
        "use_erg_tag": "use_erg_tag",
        "use_erg_lyric": "use_erg_lyric",
        "use_erg_diffusion": "use_erg_diffusion",
        "oss_steps": "oss_steps",
        "guidance_scale_text": "guidance_scale_text",
        "guidance_scale_lyric": "guidance_scale_lyric",
        "format": "format",
        "batch_size": "batch_size",
        "audio2audio_enable": "audio2audio_enable",
        "ref_audio_strength": "ref_audio_strength",
        "ref_audio_input": "ref_audio_input",
        "lora_name_or_path": "lora_name_or_path",
        "lora_weight": "lora_weight",
        "retake_seeds": "retake_seeds",
        "retake_variance": "retake_variance",
        "task": "task",
        "repaint_start": "repaint_start",
        "repaint_end": "repaint_end",
        "src_audio_path": "src_audio_path",
        "edit_target_prompt": "edit_target_prompt",
        "edit_target_lyrics": "edit_target_lyrics",
        "edit_n_min": "edit_n_min",
        "edit_n_max": "edit_n_max",
        "edit_n_avg": "edit_n_avg"
    }
    
    kwargs = {}
    for input_key, pipeline_key in pipeline_params.items():
        if input_key in validated_input:
            kwargs[pipeline_key] = validated_input[input_key]
    
    # Special handling for oss_steps with heun scheduler
    # Heun scheduler doesn't support custom sigmas, so we remove oss_steps
    if kwargs.get("scheduler_type") == "heun" and "oss_steps" in kwargs:
        if kwargs["oss_steps"] and kwargs["oss_steps"] != "":
            print(f"Warning: oss_steps not supported with heun scheduler, ignoring: {kwargs['oss_steps']}")
        kwargs.pop("oss_steps", None)
    
    # Handle double_condition cfg_type
    # If using double_condition without proper guidance scales, switch to cfg
    if kwargs.get("cfg_type") == "double_condition":
        text_scale = kwargs.get("guidance_scale_text", 0.0)
        lyric_scale = kwargs.get("guidance_scale_lyric", 0.0)
        if text_scale <= 1.0 or lyric_scale <= 1.0:
            print(f"Warning: double_condition requires guidance_scale_text > 1.0 and guidance_scale_lyric > 1.0")
            print(f"Current values: text={text_scale}, lyric={lyric_scale}. Switching to 'cfg' type.")
            kwargs["cfg_type"] = "cfg"
    
    return kwargs