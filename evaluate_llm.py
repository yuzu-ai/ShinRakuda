"""End to end evaluation script of LLMs using Shin Rakuda."""

import os
import sys

# Add the parent directory to the sys path to import rakuda package.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

from rakuda.rakuda import shin_rakuda_eval, shin_rakuda_judgement
from rakuda.config_validator import validate_config


async def eval_and_judge(config: DictConfig):
    """Evaluate and judge LLM using Shin Rakuda."""
    result_dir = await shin_rakuda_eval(config)
    await shin_rakuda_judgement(config, result_dir)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    """Evaluate LLM using Shin Rakuda."""
    print("Start Shin Rakuda evaluation...")
    try:
        # Convert Hydra config to dict and validate
        config_dict = OmegaConf.to_container(config, resolve=True)
        validated_config = validate_config(config_dict)
        print("Configuration validated successfully.")

        # Convert back to DictConfig for compatibility with existing code
        config = OmegaConf.create(validated_config.model_dump())

        asyncio.run(eval_and_judge(config))
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
