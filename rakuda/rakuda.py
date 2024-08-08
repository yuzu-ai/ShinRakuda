"""Main module for shin-rakuda evaluation."""

import datetime
import gc
import json
import os
import logging
from typing import List

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.asyncio import tqdm

from rakuda.helper import (
    bootstrap_percentiles,
    create_judge_prompt,
    generate_full_prompt,
    generate_llm_response,
    generate_models_bar_chat,
    generate_radar_chart,
    init_local_model,
    load_datasets_jsonl,
    parse_score_from_llm_response,
    save_datasets_jsonl,
    process_dataset_model_pair,
    read_jsonl,
    write_jsonl,
)
from rakuda.constants import DEFAULT_BOOTSTRAP_ITERATIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def shin_rakuda_eval(config: DictConfig):
    """Main function for shin-rakuda evaluation."""
    # Loading config YAML file
    try:
        config_dict = OmegaConf.to_container(config, resolve=True)
        assert isinstance(config_dict, dict)
    except AssertionError as exc:
        logging.error("Config is not a valid dictionary")
        raise ValueError("Invalid configuration") from exc

    inference_library = config.get("inference_library", "huggingface")

    # Check existing_eval directory
    existing_eval_dir = config.get("existing_eval_dir", None)

    if existing_eval_dir is not None:
        result_dir = f"{config.result_dir}/{existing_eval_dir}"
    else:
        result_dir = f"{config.result_dir}/{datetime.datetime.now()}"

    try:
        os.makedirs(result_dir, exist_ok=True)
    except OSError as e:
        logging.error("Failed to create result directory: %s", e)
        raise

    # Loop through each dataset and model
    for dataset in tqdm(
        config_dict["eval_datasets"], desc="Processing datasets", unit="dataset"
    ):
        print(f"Evaluating {dataset['dataset_name']}...")
        dataset_path = os.path.join(
            config.eval_datasets_dir, f"{dataset['dataset_name']}.jsonl"
        )

        if not os.path.exists(dataset_path):
            print(f"{dataset_path} does not exist.")
            continue

        input_column = dataset.get("input_column", None)
        num_questions = dataset.get("num_questions", 0)
        random_questions = dataset.get("random_questions", None)

        queries = load_datasets_jsonl(
            random_questions, num_questions, dataset_path, input_column
        )

        # save selected queries as dataset jsonl for future reference
        save_datasets_jsonl(result_dir, queries, dataset["dataset_name"])
        # loop through each model for evaluation
        for model in tqdm(
            config_dict["models"], desc="Processing models", unit="model"
        ):
            print("Evaluating model: ", model["model_name"])

            eval_result_path, file_exists = process_dataset_model_pair(
                dataset, model, config, result_dir, ""
            )
            if file_exists:
                continue

            # save the full prompt for evaluation purpose
            full_prompt_path = os.path.join(
                result_dir,
                f"{dataset['dataset_name']}++{model['model_name']}_full_prompt.jsonl",
            )

            # Initialize the local model first
            provider = model.get("provider", "huggingface").strip().lower()
            if provider == "huggingface" or provider == "hf":
                local_pipeline = init_local_model(model, inference_library)
            else:
                local_pipeline = None
            # Make sure all the turns are being considered
            MAX_CONVERSATION_TURNS = max(len(query["turns"]) for query in queries)

            chats = [[] for _ in queries]
            full_prompts = [[] for _ in queries]  # store the full prompt for each query

            for turn_idx in range(MAX_CONVERSATION_TURNS):
                query_indices = []
                prompt_indices = []
                llm_coroutines = []
                for query_idx, query in enumerate(queries):
                    if turn_idx < len(query["turns"]):
                        # Get the user content for this turn
                        content = query["turns"][turn_idx]

                        chats[query_idx].append({"role": "user", "content": content})
                        query_indices.append(query_idx)

                        prompt_indices.append(
                            generate_full_prompt(
                                local_pipeline=local_pipeline,
                                model_info=model,
                                chat_prompts=chats[query_idx],
                                inference_library=inference_library,
                            )
                        )

                        llm_coroutines.append(
                            # passing local model to generate_llm_response
                            generate_llm_response(
                                model=model,
                                local_model_pipeline=local_pipeline,
                                chat_prompts=chats[query_idx],
                                config=config_dict,
                            )
                        )

                    full_prompts[query_idx] = prompt_indices

                print(f"Processing {len(llm_coroutines)} queries...")
                responses = await tqdm.gather(*llm_coroutines)

                for query_idx, llm_response in zip(query_indices, responses):
                    chats[query_idx].append(
                        {"role": "assistant", "content": llm_response}
                    )

            write_jsonl(full_prompt_path, full_prompts)

            write_jsonl(eval_result_path, chats)

            # destroy the local model and free up memory (vllm 0.4.0.post1)
            # this method only works for vllm ver <= 0.4.0.post1
            if inference_library == "vllm":
                del local_pipeline.llm_engine.model_executor
            del local_pipeline
            gc.collect()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            ray.shutdown()

    # pass the result directory to the next function
    return result_dir


async def shin_rakuda_judgement(config: DictConfig, result_dir):
    """Load results and judge the responses"""
    try:
        config_dict = OmegaConf.to_container(config, resolve=True)
        assert isinstance(config_dict, dict)
    except AssertionError as exc:
        logging.error("Config is not a valid dictionary")
        raise ValueError("Invalid configuration") from exc

    # load the dataset jsonl file
    # based on the dataset and models, load the corresponding jsonl file
    dataset_result = []

    for dataset in tqdm(
        config_dict["eval_datasets"], desc="Processing datasets", unit="dataset"
    ):

        # check to see if dataset jsonl file exists in result dir
        result_dataset_path = os.path.join(
            result_dir, f"{dataset['dataset_name']}.jsonl"
        )

        if not os.path.exists(
            os.path.join(result_dir, f"{dataset['dataset_name']}.jsonl")
        ):
            # load the dataset jsonl file
            result_dataset_path = os.path.join(
                config.eval_datasets_dir, f"{dataset['dataset_name']}.jsonl"
            )
            if not os.path.exists(result_dataset_path):
                print(f"{result_dataset_path} does not exist.")
                continue

        input_column = dataset.get("input_column", None)

        queries = load_datasets_jsonl(None, 0, result_dataset_path, input_column)

        for model in tqdm(
            config_dict["models"], desc="Processing models", unit="model"
        ):
            # load the jsonl file
            result_path = os.path.join(
                result_dir,
                f"{dataset['dataset_name']}++{model['model_name']}.jsonl",
            )
            try:
                results = read_jsonl(result_path)
            except FileNotFoundError:  # if the file does not exist
                print(f"{result_path} does not exist.")
                continue

            # create a new file for judged results
            judged_result_path, file_exists = process_dataset_model_pair(
                dataset, model, config, result_dir, "_judged"
            )

            # check if the judge file already exists
            if config.get("existing_eval", False) and file_exists:
                print(f"{judged_result_path} already exists.")
                judged_result = True
            else:
                judged_result = False

            if not judged_result:
                scores: List = []
                judge_coroutines = []
                for i, result in tqdm(
                    enumerate(results), desc="Judging results", unit="result"
                ):
                    # result has all the interaction between the user and the model

                    # create a judge prompt and have the judge model score the response
                    judge_prompt = create_judge_prompt(
                        query=queries[i],
                        messages=result,
                        judge_prompt=dataset["judge_prompt_template"],
                        use_jinja=dataset.get("use_jinja", False),
                    )
                    # the judge model doesn't work with local model yet

                    judge_coroutines.append(
                        generate_llm_response(
                            model=config.judge_models[0],
                            local_model_pipeline=None,
                            chat_prompts=judge_prompt,
                            config=config,
                        )
                    )

                    query_category = queries[i].get("category", None)

                    scores.append(
                        {
                            "query": judge_prompt,
                            "category": query_category,
                        }
                    )

                responses = await tqdm.gather(*judge_coroutines)

                for i, judge_response in enumerate(responses):
                    scores[i]["score"] = parse_score_from_llm_response(
                        judge_response, dataset["score_keyword"]
                    )
                    scores[i]["response"] = str(judge_response)

                # save this model's results to a json file
                write_jsonl(judged_result_path, scores)
            else:
                # load the judged results
                scores = read_jsonl(judged_result_path)

            mean_score = sum([score["score"] for score in scores]) / len(scores)

            n_bootstrap = DEFAULT_BOOTSTRAP_ITERATIONS

            print(
                f"Running {n_bootstrap} bootstrap samples for {dataset['dataset_name']} {model['model_name']}"
            )
            confidence_region = bootstrap_percentiles(
                scores,
                lambda x: sum([score["score"] for score in x]) / len(x),
                percentiles=[2.5, 97.5],
                n_bootstrap=n_bootstrap,
            )

            # generate radar graph if multiple categories
            generate_radar_chart(
                scores=scores,
                model_name=model["model_name"],
                dataset_name=dataset["dataset_name"],
                judge_model=config.judge_models[0],
                result_dir=result_dir,
            )

            dataset_result.append(
                {
                    "dataset": dataset,
                    "model": model["model_name"],
                    "result_path": result_path,
                    "score": mean_score,
                    "95_confidence_lower": confidence_region[0],
                    "95_confidence_upper": confidence_region[1],
                }
            )

    # Generate a overall graph with all the scores, based on each dataset with models
    generate_models_bar_chat(all_scores=dataset_result, result_dir=result_dir)

    # save all the result
    final_result_path = os.path.join(
        config.result_dir,
        f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    write_jsonl(final_result_path, dataset_result)
