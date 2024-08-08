# config_validator.py
from typing import List, Optional, Any
from pydantic import BaseModel, validator, ValidationError


class VLLMConfig(BaseModel):
    model: str
    max_model_len: int


class VLLMSamplingParams(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float


class HFPipeline(BaseModel):
    task: str
    model: str
    torch_dtype: str
    max_new_tokens: int
    device_map: str
    trust_remote_code: bool
    return_full_text: bool


class HFChatTemplate(BaseModel):
    chat_template: str
    tokenize: bool
    add_generation_prompt: bool


class Model(BaseModel):
    model_name: str
    api: bool = True
    provider: str
    system_prompt: Optional[str]
    do_sample: Optional[bool]
    vllm_config: Optional[VLLMConfig]
    vllm_sampling_params: Optional[VLLMSamplingParams]
    hf_pipeline: Optional[HFPipeline]
    hf_chat_template: Optional[HFChatTemplate]

    @validator("provider")
    def validate_provider(cls, v: str) -> str:
        allowed_providers = ["openai", "anthropic", "google", "huggingface", "hf"]
        if v.lower() not in allowed_providers:
            raise ValueError(f"Provider must be one of {allowed_providers}")
        return v.lower()


class Dataset(BaseModel):
    dataset_name: str
    judge_prompt_template: str
    num_questions: Optional[int]
    random_questions: Optional[bool]
    use_jinja: bool
    score_keyword: str


class JudgeModel(BaseModel):
    model_name: str
    api: bool
    provider: str


class Config(BaseModel):
    models: List[Model]
    eval_datasets: List[Dataset]
    judge_models: List[JudgeModel]
    eval_datasets_dir: str
    log_dir: str
    result_dir: str
    existing_eval_dir: Optional[str]
    inference_library: str

    @validator("inference_library")
    def validate_inference_library(cls, v: str) -> str:
        allowed_libraries = ["vllm", "hf", "huggingface"]
        if v.lower() not in allowed_libraries:
            raise ValueError(f"Inference library must be one of {allowed_libraries}")
        return v.lower()


def format_error_message(e: ValidationError) -> str:
    """Format validation error messages with more detail."""
    error_messages = []
    for error in e.errors():
        location = " -> ".join(str(loc) for loc in error["loc"])
        message = f"Error in {location}: {error['msg']}"
        error_messages.append(message)
    return "\n".join(error_messages)


def validate_config(config_dict: dict) -> Config:
    try:
        return Config(**config_dict)
    except ValidationError as e:
        detailed_error = format_error_message(e)
        raise ValueError(f"Configuration validation failed:\n{detailed_error}")


def validate_nested_config(config: Any, path: str = "") -> List[str]:
    """Recursively validate nested configuration and return a list of error messages."""
    errors = []
    if isinstance(config, BaseModel):
        try:
            config.model_validate({})
        except ValidationError as e:
            for error in e.errors():
                loc = " -> ".join(str(l) for l in error["loc"])
                full_path = f"{path}.{loc}" if path else loc
                errors.append(f"Error in {full_path}: {error['msg']}")
    elif isinstance(config, dict):
        for key, value in config.items():
            new_path = f"{path}.{key}" if path else key
            errors.extend(validate_nested_config(value, new_path))
    elif isinstance(config, list):
        for i, item in enumerate(config):
            new_path = f"{path}[{i}]"
            errors.extend(validate_nested_config(item, new_path))
    return errors


def deep_validate_config(config_dict: dict) -> Config:
    """Perform deep validation of the configuration."""
    try:
        config = Config.model_validate(config_dict)
        errors = validate_nested_config(config)
        if errors:
            raise ValueError(
                "Nested configuration validation failed:\n" + "\n".join(errors)
            )
        return config
    except ValidationError as e:
        detailed_error = format_error_message(e)
        raise ValueError(f"Configuration validation failed:\n{detailed_error}")
