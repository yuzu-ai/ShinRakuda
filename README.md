# Shin Rakuda

English | [日本語](README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Table of Contents
- [Shin Rakuda](#shin-rakuda)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Key Features](#key-features)
  - [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
    - [Models](#models)
    - [Evaluation Datasets](#evaluation-datasets)
    - [Judge Model](#judge-model)
    - [Evaluation Configurations](#evaluation-configurations)
  - [Installation](#installation)
    - [Alternative Dependency Management](#alternative-dependency-management)
  - [Usage](#usage)
  - [Todo](#todo)
  - [Contributing](#contributing)
  - [Troubleshooting](#troubleshooting)
  - [License](#license)
  - [Citation](#citation)
  - [References](#references)

## Description

Shin Rakuda is a powerful and flexible tool designed to benchmark the performance of different Language Models (LLMs) on given datasets. It provides researchers and developers with an easy-to-use interface to load datasets, select models, run benchmarking processes, and visualize results.

## Key Features

- Support for multiple inference libraries (Huggingface and VLLM)
- Flexible configuration for models, datasets, and evaluation parameters
- Easy-to-use command-line interface
- Visualization of benchmarking results
- Support for both API-based and local models

## Prerequisites

- Python 3.9 or higher
- pip or Poetry for dependency management
- Access to required model APIs (if using API-based models)
- Sufficient computational resources for running local models (if applicable)

## Configuration

1. Copy the `.env.example` file to `.env` and configure the models' API keys if necessary:

   ```bash
   cp .env.example .env
   ```

2. Edit the `config.yaml` file to configure the project. The configuration file is divided into several sections:

   - Models: Define the LLMs you want to benchmark
   - Evaluation Datasets: Specify the datasets for evaluation
   - Judge Model: Configure the model used for judging responses
   - Evaluation Configurations: Set up directories and other evaluation parameters

   For detailed explanations of each configuration option, please refer to the comments in the `config_template.yaml` file.

### Models

```yaml
# API model
models:
  - model_name: string
    api: boolean # whether the model inference via API, default True for API models
    provider: string # the provider of the model
# Local Model
  - model_name: string # can be any name you want
    api: boolean # whether the model inference via API, default False for local models
    provider: string # model hosting provider, default huggingface
    system_prompt: string 
    do_sample: boolean
    vllm_config: # vllm config section
      model: string # model full name or model id
      max_model_len: int # maximum model length
    vllm_sampling_params: # vllm sampling parameters section
      temperature: float # temperature
      top_p: float # top p
      max_tokens: int # maximum tokens
      repetition_penalty: float # repetition penalty
    hf_pipeline: # huggingface pipeline section
      task: string
      model: string # model full name or model id
      torch_dtype: string 
      max_new_tokens: int
      device_map: string
      trust_remote_code: boolean
      return_full_text: boolean
    hf_chat_template: # huggingface chat template section
      chat_template: string # this can be either the complete chat template or the format of the chat template such as `ChatML`
      tokenize: boolean # whether to tokenize the chat template
      add_generation_prompt: boolean # whether to add generation prompt
```

References:

- [VLLM Engine](https://docs.vllm.ai/en/latest/dev/engine/llm_engine.html#vllm.LLMEngine)
- [VLLM Sampling Parameters](https://docs.vllm.ai/en/latest/dev/sampling_params.html)
- [Huggingface Pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline)
- [Huggingface Chat Template](https://huggingface.co/docs/transformers/en/chat_templating)

Please add the HF or VLLM configuration parameters as you see fit, and Rakuda will process accordingly. Rakuda will NOT work if you have extra parameters that are not supported by either inference library.

### Evaluation Datasets

```yaml
eval_datasets:
  - dataset_name: string # dataset name
    judge_prompt_template: string # judge prompt template
    num_questions: int # Optional, number of questions to evaluate
    random_questions: boolean # Optional, whether to select questions randomly when num_questions is provided
    use_jinja: boolean # whether to use jinja templating for the judge prompt
    score_keyword: string # keyword to extract the score from the model output, please see config_template.yaml file for format
```

### Judge Model

```yaml
judge_models:
  - model_name: string # repo_id of model
    api: boolean # whether the model inference via API
    provider: string # the provider of the model
```

### Evaluation Configurations

```yaml
eval_datasets_dir: string # directory containing the evaluation datasets
log_dir: string # directory to save the logs
result_dir: string # directory to save the evaluation results
existing_eval_dir: string  # Optional, directory containing existing results to compare with, so it will not re-run the evaluation for some models
inference_library: string  # inference library to use, change to "hf" or "huggingface" for huggingface, "vllm" for vllm
```

## Installation

```bash
# Create a virtual environment
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the dependencies
pip install -r requirements.txt
# Update filelock to resolve bug
pip install --upgrade filelock
```

### Alternative Dependency Management

This project uses `pyproject.toml` for dependency management. To install additional dependencies:

1. Add the dependency to the `pyproject.toml` file.

```bash
poetry add <dependency>
```

2. Run `poetry install` to update your environment.

## Usage

Run the end-to-end evaluation script:

```bash
python3 scripts/evaluate_llm.py --config-name config_xxx
```

Replace `config_xxx` with the name of your configuration file (without .yaml) located in the `configs` directory.

Example output:
```
Start Shin Rakuda evaluation...
Processing datasets: 100%|██████████| 2/2 [00:00<00:00,  5.01it/s]
Evaluating japanese_mt_bench...
Processing models: 100%|██████████| 3/3 [00:00<00:00, 15.08it/s]
...
```

After the evaluation is complete, you can find the results and visualizations in the `result_dir` specified in your configuration file.

## Todo

- [ ] Add support for llama 3.1 models
- [ ] Improve Huggingface pipeline support
- [ ] Update VLLM (until vllm latest version that supports proper gpu memory release)

## Contributing

We welcome contributions to Shin Rakuda! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the project's coding standards.

## Troubleshooting

- If you encounter CUDA out of memory errors, try reducing the `max_model_len` or `max_tokens` parameters in your model configuration.
- For issues with specific models or datasets, check the model provider's documentation or dataset source for any known limitations or requirements.
- If you're having trouble with dependencies, make sure you're using the correct version of Python and have installed all required packages.

For more help, please open an issue on the GitHub repository.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation

If you use Shin Rakuda in your research, please cite it as follows:

```bibtex
@software{shin_rakuda,
  author = {YuzuAI},
  title = {Shin Rakuda: A Flexible LLM Benchmarking Tool},
  year = {2024},
  url = {https://github.com/yourusername/shin-rakuda}
}
```

## References
