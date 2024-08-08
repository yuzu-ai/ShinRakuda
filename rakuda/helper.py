"""Helper functions for Shin Rakuda."""

import json
import logging
import os
import re
from collections import defaultdict
from math import pi
from random import choices, sample
from typing import Optional, Dict, List

import anthropic
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import openai
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import pipeline, AutoTokenizer
from vllm import SamplingParams, LLM

from rakuda.constants import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_PERCENTILES,
    DEFAULT_FIG_WIDTH,
    DEFAULT_BASE_FONT_SIZE,
    DEFAULT_BAR_CHART_WIDTH,
    DEFAULT_BAR_CHART_HEIGHT
)

# Load the .env file and export all the keys as environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def write_jsonl(file_path: str, data: List[Dict]):
    """
    Write a list of dictionaries to a JSON Lines file.

    Args:
        file_path (str): The path to the output file.
        data (List[Dict]): The list of dictionaries to be written.

    Returns:
        None
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a JSON object from the file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return [json.loads(line) for line in f]
        except json.JSONDecodeError as e:
            logging.error("Error decoding JSON: %s", e)
            return []


def parse_float(string: str) -> float:
    """Clean up string using regex and convert to float."""
    string = re.sub(r"[^\d.]", "", string)
    try:
        return float(string)
    except ValueError:
        return 0.0


def bootstrap_percentiles(
    scores, score_function, percentiles=None, n_bootstrap=DEFAULT_BOOTSTRAP_ITERATIONS
):
    """Calculate the bootstrap percentiles."""
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    try:
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = choices(np.arange(len(scores)), k=len(scores))
            bs_scores = [scores[idx] for idx in indices]
            bootstrap_scores.append(score_function(bs_scores))

        return np.percentile(bootstrap_scores, percentiles)
    except Exception as e:
        logging.error("Error in bootstrap percentiles calculation: %s", e)
        raise


def format_chat_template(chat_template):
    """Format the chat template."""
    return chat_template.replace("    ", "").replace("\n", "")


def parse_score_from_llm_response(llm_response: str, match_keyword: str) -> int:
    """Parse the score from the LLM response."""
    match = re.search(rf"{match_keyword}", llm_response)

    if match:
        score = match.group(1)
        # print(f"Extracted score: {score}")
        return int(score)
    else:
        score = re.findall(r"(\d+)", llm_response)
        if score:
            return int(score[0])
    return 0


def process_dataset_model_pair(dataset, model, config, result_dir, operation):
    """Check if the result file already exists and return the path."""
    result_path = os.path.join(
        result_dir,
        f"{dataset['dataset_name']}++{model['model_name']}{operation}.jsonl",
    )

    existing_eval_dir = config.get("existing_eval_dir", None)
    if existing_eval_dir and os.path.exists(result_path):
        print(f"{result_path} already exists.")
        return result_path, True  # Indicate that the file already exists

    return result_path, False


def create_judge_prompt(
    query,
    judge_prompt: str,
    messages: list,
    use_jinja: bool = False,
) -> str:
    """Create a judge prompt with .format()"""

    if use_jinja:
        template = Template(judge_prompt)
        return template.render(messages=messages, query=query)
    else:
        question_count, answer_count = 1, 1

        for _, message in enumerate(messages):
            if message["role"] != "system":
                if message["role"] == "user":
                    key = f"question{question_count}"
                    question_count += 1
                elif message["role"] == "assistant":
                    key = f"answer{answer_count}"
                    answer_count += 1

                # Add the item content to the dictionary
                query[key] = message["content"]

        return judge_prompt.format(**query)


def init_local_model(model_info, inference_library: str) -> pipeline:
    """Initialize the local model."""
    print(f"\n*** Initializing local model {model_info['model_name']} ***\n")
    try:
        if inference_library == "vllm":
            pipe = LLM(**model_info["vllm_config"])
        elif inference_library in ["hf", "huggingface"]:
            pipe = pipeline(**model_info["hf_pipeline"])
        else:
            raise ValueError(f"Inference library '{inference_library}' is not supported.")

        return pipe
    except Exception as e:
        logging.error("Failed to initialize local model: %s", e)
        raise


def generate_local_model_prompt(
    local_pipeline,
    model_info,
    chat_prompts,
    inference_library: str,
    return_text: bool = False,
):
    """Generate prompt for local models."""

    # chat_prompts should be a list of dictionaries with role and content keys
    if isinstance(chat_prompts, str):
        chat_prompts = [{"role": "user", "content": chat_prompts}]

    system_prompt = model_info.get("system_prompt", None)
    system_prompt_included = False

    for chat in chat_prompts:
        if chat["role"] == "system":
            system_prompt_included = True
            break

    if not system_prompt_included and system_prompt:
        chat_prompts.insert(0, {"role": "system", "content": system_prompt})

    chat_template = model_info["hf_chat_template"].get("chat_template", None)

    if inference_library in ["hf", "huggingface"]:
        """Generate prompt for local models."""
        tokenizer = local_pipeline.tokenizer
        if chat_template:
            chat_template = format_chat_template(chat_template)
        else:
            chat_template = tokenizer.default_chat_template

        # update chat template option
        chat_template_options = model_info["hf_chat_template"]
        chat_template_options["chat_template"] = chat_template
        if return_text:  # override the tokenize value if return_text is True
            chat_template_options["tokenize"] = False

        chat_prompts = tokenizer.apply_chat_template(
            chat_prompts,
            **chat_template_options,
        )
    elif inference_library == "vllm":
        if not chat_template:
            chat_template = """
            {{ system_prompt }}
            {% for message in conversation %}
            {{ 'USER:' if message.role == 'user' else 'ASSISTANT:' }} {{ message.content }}
            {% endfor %}
            """
        elif chat_template.lower() == "chatml":
            tokenizer = AutoTokenizer.from_pretrained(
                model_info["vllm_config"]["model"]
            )
            chat_template = tokenizer.default_chat_template

        chat_template = format_chat_template(chat_template)
        template = Template(chat_template)
        bos_token = model_info.get("bos_token", "")
        eos_token = model_info.get("eos_token", "")
        chat_prompts = template.render(
            messages=chat_prompts,
            system_prompt=system_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    print("*** Local model prompt: ", chat_prompts)

    return chat_prompts


def generate_full_prompt(
    local_pipeline,
    model_info,
    chat_prompts,
    inference_library,
    return_text: bool = False,
):
    """Generate a full prompt for the model for review purpose"""
    if isinstance(chat_prompts, str):
        chat_prompts = [{"role": "user", "content": f"{chat_prompts}"}]

    provider = model_info["provider"].strip().lower()
    # generate a full prompt text for review purpose
    if not local_pipeline:  # generate prompt for api models
        if provider == "openai":
            return chat_prompts
        elif provider == "google":
            chat_prompt_text = " ".join([x["content"] for x in chat_prompts])
            return chat_prompt_text
        elif provider == "anthropic":
            return chat_prompts
    else:
        chat_prompt_text = generate_local_model_prompt(
            local_pipeline, model_info, chat_prompts, inference_library, return_text
        )
        return chat_prompt_text


@retry(
    retry=retry_if_exception_type(
        (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APIError,
            anthropic.RateLimitError,
            anthropic.APIError,
            anthropic.APIConnectionError,
        )
    ),
    wait=wait_random_exponential(multiplier=1, min=1, max=100),
    stop=stop_after_attempt(5),
)
async def generate_llm_response(
    model,
    config: Dict,
    local_model_pipeline: Optional[pipeline],
    chat_prompts: list | str,
) -> str:
    """Get the response from the LLM."""

    # if the chat prompt is a string, convert it to a list, mostly for judge prompt

    inference_library = config.get("inference_library", "hf")

    if isinstance(chat_prompts, str):
        chat_prompts = [{"role": "user", "content": f"{chat_prompts}"}]

    try:
        provider = model["provider"].strip().lower()

        if provider == "openai":
            openai_client = AsyncOpenAI()

            llm_response = await openai_client.chat.completions.create(
                model=model["model_name"], messages=chat_prompts
            )
            response = llm_response.choices[0].message.content
        elif provider == "google":
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=GOOGLE_API_KEY)
            google_client = genai.GenerativeModel("gemini-pro")
            chat_prompt_text = " ".join([x["content"] for x in chat_prompts])
            response = google_client.generate_content(chat_prompt_text)
            response = response.text
        elif provider == "anthropic":
            anthropic_client = AsyncAnthropic()
            message = await anthropic_client.messages.create(
                model=model["model_name"], messages=chat_prompts, max_tokens=1024
            )
            response = message.content
        elif provider == "huggingface":
            if inference_library == "vllm":
                return_text = True
            else:
                return_text = False  # TODO: need to check if this is necessary for HF

            chat_prompt_text = generate_local_model_prompt(
                local_model_pipeline, model, chat_prompts, inference_library, return_text
            )

            if inference_library in ["hf", "huggingface"]:
                do_sample_option = model.get("do_sample", True)
                outputs = local_model_pipeline(
                    chat_prompt_text,
                    do_sample=do_sample_option,
                    pad_token_id=local_model_pipeline.tokenizer.eos_token_id,
                )
                response = outputs[0]["generated_text"]
            elif inference_library == "vllm":

                sampling_params = SamplingParams(
                    **model["vllm_sampling_params"]
                )

                output = local_model_pipeline.generate(chat_prompt_text, sampling_params)
                response = output[0].outputs[0].text
            else:
                raise ValueError(f"Inference library '{inference_library}' is not supported.")
        else:
            raise ValueError(f"Model provider '{model['provider']}' is not supported.")
    except KeyError:
        logging.error("Missing 'provider' key in model configuration")
        raise
    except Exception as e:
        logging.error("Error generating LLM response: %s", e)
        raise

    response = str(response).strip()
    print(f"Response: {response}")
    return response


def load_datasets_jsonl(
    random_questions: Optional[bool],
    num_questions: int,
    dataset_path: str,
    column: Optional[str],
) -> list:
    """Load the dataset from the given path."""

    queries = []
    try:
        queries = read_jsonl(dataset_path)
    except FileNotFoundError:
        logging.error("Dataset file not found: %s", dataset_path)
        return []

    # due to datasets format, we need to load the turns differently
    # No changes needed for datasets like "japanese-mt-bench" with "turns" key

    if column:
        queries = [{**query, "turns": [query[column]]} for query in queries]

    # if random_questions is None, as key is not in config, then no shuffle
    if random_questions is not None:
        if num_questions > 0 and random_questions:
            queries = sample(queries, num_questions)
        elif num_questions > 0 and not random_questions:
            queries = queries[:num_questions]
    else:
        if num_questions > 0:
            queries = queries[:num_questions]

    return queries


def save_datasets_jsonl(dataset_path: str, queries: list, dataset_name: str):
    """Save the dataset to the given path."""
    dataset_path = os.path.join(dataset_path, f"{dataset_name}.jsonl")

    write_jsonl(dataset_path, queries)


def generate_radar_chart(
    scores: list, model_name: str, dataset_name: str, judge_model: dict, result_dir: str
):
    """Generate a graph from the given scores."""

    try:
        fig_width = DEFAULT_FIG_WIDTH  # Example figure width in inches
        base_font_size = DEFAULT_BASE_FONT_SIZE  # Base font size for a figure of default size (6x6 inches)

        title_font_size = base_font_size + (fig_width - 6)
        subtitle_font_size = base_font_size + (fig_width - 6) * 0.5

        categories = set(score["category"] for score in scores)
        num_categories = len(categories)

        # Calculate the mean scores for each category
        mean_scores = {}
        for category in categories:
            category_scores = [
                score["score"] for score in scores if score["category"] == category
            ]
            mean_scores[category] = sum(category_scores) / len(category_scores)
        # if there are multiple categories, create spider web graph

        labels, values = zip(*mean_scores.items())
        values = list(values)
        labels = list(labels)

        if num_categories > 1:
            angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
            angles += angles[:1]

            values += values[:1]  # Complete the loop

            # Set up the figure and polar subplot
            fig, ax = plt.subplots(
                figsize=(num_categories, num_categories), subplot_kw=dict(polar=True)
            )

            # Draw one axe per variable and add labels
            plt.xticks(angles[:-1], labels)

            # Plot data and fill with color
            ax.plot(angles, values, linewidth=1, linestyle="solid")
            ax.fill(angles, values, "b", alpha=0.25)

            # Add a title and a grid
            plt.title(f"{dataset_name} - {model_name}", size=title_font_size)
            plt.suptitle(
                f"Judge: {judge_model['model_name']}", size=subtitle_font_size, y=0.05
            )
            ax.grid(True)

            # Save the plot
            file_path = (
                f"{result_dir}/radar_chart_{dataset_name}_{model_name}_{judge_model}.png"
            )
            plt.savefig(file_path, bbox_inches="tight")

            return file_path
    except Exception as e:
        logging.error("Failed to generate radar chart: %s", e)
        return None


def generate_vertical_bar_chart(dataset_scores: list, dataset: str, result_dir: str):
    """Generate a vertical bar chart from the given scores."""

    try:
        # Extract the model names and scores from the dataset scores
        scores = [score["score"] for score in dataset_scores]
        # dataset_scores = json.load(dataset_scores)
        models = [score["model"] for score in dataset_scores]

        # Plotting
        fig, ax = plt.subplots(
            figsize=(DEFAULT_BAR_CHART_WIDTH, DEFAULT_BAR_CHART_HEIGHT)
        )

        # Create the vertical bar chart
        bars = ax.bar(models, scores, color="grey")  # Default color is grey

        # Highlight a specific bar (e.g., the model ELYZA-7B) in blue
        # Add the score labels above each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        # Add titles and labels
        ax.set_title(f"Model Benchmark for Dataset: {dataset}")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")

        # Rotate the model names for better readability
        plt.xticks(rotation=45)

        # Save the plot to disk
        file_path_vertical = f"{result_dir}/vertical_bar_chart_{dataset}.png"
        plt.savefig(file_path_vertical, bbox_inches="tight", format="png")

        # Return the path of the saved file
        return file_path_vertical
    except Exception as e:
        logging.error("Failed to generate vertical bar chart: %s", e)
        return None


def generate_models_bar_chat(all_scores: list, result_dir: str):
    """Generate one bar chart for each dataset with all the models."""
    scores_by_dataset = defaultdict(list)

    for score in all_scores:
        # Use dictionary comprehension to exclude the 'dataset' key
        reduced_score = {k: v for k, v in score.items() if k != "dataset"}
        scores_by_dataset[score["dataset"]["dataset_name"]].append(reduced_score)

    # # Convert defaultdict to a regular dict if necessary
    scores_by_dataset = dict(scores_by_dataset)

    # To get a list of lists of the reduced objects
    datasets = list(scores_by_dataset.keys())

    chart_results = []

    for dataset in datasets:
        chart_results.append(
            generate_vertical_bar_chart(
                dataset_scores=scores_by_dataset[dataset],
                dataset=dataset,
                result_dir=result_dir,
            )
        )

    return chart_results
