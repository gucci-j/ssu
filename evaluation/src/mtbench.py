# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401, I001
import re
import numpy as np
from pydantic import BaseModel
from typing import Callable, Literal, Literal
from huggingface_hub import HfApi

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping, MetricCategory, MetricUseCase
from lighteval.tasks.extended.mt_bench.judge_prompt_templates import (
    flow_judge_prompt_mt_bench_with_ref,
    flow_judge_prompt_mt_bench_without_ref,
)

### Source from llm_as_judge.py ###
DEFAULT_FORMAT = {"type": "text"}
class JudgeLM:
    """
    A class representing a judge for evaluating answers using either the OpenAI or Transformers library.

    Args:
        model (str): The name of the model.
        templates (Callable): A function taking into account the question, options, answer, and gold and returning the judge prompt.
        process_judge_response (Callable): A function for processing the judge's response.
        judge_backend (Literal["openai", "transformers", "tgi", "vllm"]): The backend for the judge.
        url (str | None): The URL for the OpenAI API.
        api_key (str | None): The API key for the OpenAI API (either OpenAI or HF key).

    Attributes:
        model (str): The name of the model.
        template (Callable): A function taking into account the question, options, answer, and gold and returning the judge prompt.
        API_MAX_RETRY (int): The maximum number of retries for the API.
        API_RETRY_SLEEP (int): The time to sleep between retries.
        client (OpenAI | None): The OpenAI client.
        pipe (LLM | AutoModel | None): The Transformers or vllm pipeline.
        process_judge_response (Callable): A function for processing the judge's response.
        url (str | None): The URL for the OpenAI API.
        api_key (str | None): The API key for the OpenAI API (either OpenAI or HF key).
        backend (Literal["openai", "transformers", "tgi", "vllm"]): The backend for the judge

    Methods:
        evaluate_answer: Evaluates an answer using the OpenAI API or Transformers library.
        __lazy_load_client: Lazy loads the OpenAI client or Transformers pipeline.
        __call_api: Calls the API to get the judge's response.
        __call_transformers: Calls the Transformers pipeline to get the judge's response.
        __call_vllm: Calls the VLLM pipeline to get the judge's response.
    """

    def __init__(
        self,
        model: str,
        templates: Callable,
        process_judge_response: Callable,
        judge_backend: Literal["litellm", "openai", "transformers", "tgi", "vllm", "inference-providers"],
        url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        response_format: BaseModel = None,
    ):
        self.model = model
        self.template = templates

        self.API_MAX_RETRY = 3
        self.API_RETRY_SLEEP = 1

        self.client = None
        self.pipe = None
        self.process_judge_response = process_judge_response

        self.url = url
        self.api_key = api_key
        self.backend = judge_backend
        self.max_tokens = max_tokens

        self.response_format = response_format if not None else DEFAULT_FORMAT


    def __lazy_load_client(self):  # noqa: C901
        match self.backend:
            # Both "openai" and "tgi" backends use the OpenAI-compatible API
            # They are handled separately to allow for backend-specific validation and setup
            case "transformers":
                if self.pipe is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                    transformers_model = AutoModelForCausalLM.from_pretrained(
                        self.model, torch_dtype=torch.bfloat16, trust_remote_code=False, device_map="cuda"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.model)
                    self.pipe = pipeline(
                        "text-generation",
                        model=transformers_model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.8, 
                        top_p=0.95,
                    )
                return self.__call_transformers

            case _:
                raise ValueError(f"Unsupported backend: {self.backend}")

    def dict_of_lists_to_list_of_dicts(self, dict_of_lists):
        """
        Transform a dictionary of lists into a list of dictionaries.

        Each dictionary in the output list will contain one element from each list in the input dictionary,
        with the same keys as the input dictionary.

        Args:
            dict_of_lists: A dictionary where each value is a list.
                           All lists are expected to have the same length.

        Returns:
            A list of dictionaries.

        Example:
            >>> dict_of_lists_to_list_of_dicts({'k': [1, 2, 3], 'k2': ['a', 'b', 'c']})
            [{'k': 1, 'k2': 'a'}, {'k': 2, 'k2': 'b'}, {'k': 3, 'k2': 'c'}]
        """
        # Check if input is empty
        if not dict_of_lists:
            return None

        # Get all list lengths to ensure they match
        list_lengths = [len(values) for values in dict_of_lists.values()]

        # Ensure all lists have the same length
        if len(set(list_lengths)) > 1:
            raise ValueError("All lists in the input dictionary must have the same length")

        # Get the length of the lists
        n = list_lengths[0] if list_lengths else 0

        # Create list of dictionaries
        result = []
        for i in range(n):
            new_dict = {key: values[i] for key, values in dict_of_lists.items()}
            result.append(new_dict)

        return result

    def evaluate_answer_batch(
        self,
        questions: list[str],
        answers: list[str],
        options: list[list[str]] | list[None],
        golds: list[str] | list[None],
        **kwargs,
    ):
        judge_function = self.__lazy_load_client()

        kwargss = self.dict_of_lists_to_list_of_dicts(kwargs)
        if kwargss is None:
            kwargss = [{} for _ in range(len(questions))]

        # enumerate over questions answers options and golds to make the
        prompts = [
            self.template(question=q, answer=a, options=o, gold=g, **k)
            for q, a, o, g, k in zip(questions, answers, options, golds, kwargss)
        ]
        responses = judge_function(prompts)
        scores = [self.process_judge_response(response) for response in responses]

        # clean up the vllm pipeline and free up memory
        if self.pipe is not None and self.backend == "vllm":
            del self.pipe
            self.pipe = None

        return scores, prompts, responses

    def evaluate_answer(self, question: str, answer: str, options: list[str] | None = None, gold: str | None = None):
        """
        Evaluates an answer using either Transformers or OpenAI API.

        Args:
            questions (list[str]): The prompt asked to the evaluated model
            answers (list[str]): Answer given by the evaluated model
            references (list[str]): A list of reference answers

        Returns:
            A tuple containing the score, prompts, and judgment.
        """
        # lazy loading of the pipeline
        judge_function = self.__lazy_load_client()
        prompt = self.template(question=question, options=options, answer=answer, gold=gold)
        response = judge_function(prompt)
        score = self.process_judge_response(response)

        return score, prompt, response

    def __call_transformers(self, prompt):
        response = self.pipe(prompt)[0]["generated_text"]
        response = response[-1]["content"]
        return response


### Source from metrics_sample ###
class JudgeLLM:
    def __init__(
        self,
        judge_model_name: str,
        template: Callable,
        process_judge_response: Callable,
        judge_backend: Literal["transformers", "vllm"],
        short_judge_name: str | None = None,
        response_format: BaseModel = None,
        url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        api_key = None
        match judge_backend:
            case "transformers" | "vllm":
                api = HfApi()
                models = api.list_models(model_name=judge_model_name)
                if not models:
                    raise ValueError(f"{judge_model_name} not found on Hugging Face Hub")

            case _:
                raise ValueError(f"{judge_backend} is not a valid backend for llm as a judge metric")

        self.short_judge_name = short_judge_name
        self.judge = JudgeLM(
            model=judge_model_name,
            templates=template,
            process_judge_response=process_judge_response,
            judge_backend=judge_backend,
            response_format=response_format,
            api_key=api_key,
            url=url,
            max_tokens=max_tokens,
        )

    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs) -> dict[str, float]:
        raise NotImplementedError("This method should be implemented in the subclass.")


class JudgeLLMMTBench(JudgeLLM):
    def compute(self, predictions: list[str], formatted_doc: Doc, **kwargs):
        """
        Compute the score of a generative task using a llm as a judge.
        The generative task can be multiturn with 2 turns max, in that case, we
        return scores for turn 1 and 2. Also returns user_prompt and judgement
        which are ignored later by the aggregator.
        """
        import json

        # If we are evaluating a multiturn task, we need to have specific field in the formatted doc
        questions = formatted_doc.specific["multi_turn_queries"]
        golds = formatted_doc.specific.get("reference", None)

        query_context_1 = {"query": questions[0], "context": ""}
        query_context_2 = {"query": questions[1], "context": predictions[0].result[0]}
        
        score_turn_1, message_turn_1, judgement_turn_1 = self.judge.evaluate_answer(
            question=json.dumps(query_context_1, indent=2), answer=predictions[0].result[0], gold=golds[0] if golds else None
        )
        score_turn_2, message_turn_2, judgement_turn_2 = self.judge.evaluate_answer(
            question=json.dumps(query_context_2, indent=2), answer=predictions[0].result[1], gold=golds[1] if golds else None
        )

        return {
            "judge_score_turn_1": score_turn_1,
            "judge_score_turn_2": score_turn_2,
            "user_prompt": [message_turn_1, message_turn_2],
            "judgement": [judgement_turn_1, judgement_turn_2],
        }

### Following is from the original main.py file ###
def mt_bench_prompt(line, task_name: str = ""):
    return Doc(
        task_name=task_name,
        query=f"{line['turns'][0]}",
        choices=[],
        instruction=None,
        gold_index=[],
        specific={
            "reference": line["reference"],
            "category": line["category"],
            "multi_turn_queries": line["turns"],
            "id": line["question_id"],
        },
    )


def process_judge_response(x):
    search = re.search(r"<score>\s(\d)\s</score>", x)
    return int(search.group(1)) if search else 0


def flow_judge_mt_bench_prompt(question, answer, options, gold):
    if gold is not None and len(gold) > 0:
        return flow_judge_prompt_mt_bench_with_ref(question, options, answer, gold)

    return flow_judge_prompt_mt_bench_without_ref(question, options, answer, gold)


llm_judge_mt_bench = SampleLevelMetricGrouping(
    metric_name=["judge_score_turn_1", "judge_score_turn_2"],
    higher_is_better={"judge_score_turn_1": True, "judge_score_turn_2": True},
    category=MetricCategory.LLM_AS_JUDGE_MULTI_TURN,
    use_case=MetricUseCase.SUMMARIZATION,
    sample_level_fn=JudgeLLMMTBench(
        judge_model_name="flowaicom/Flow-Judge-v0.1",
        template=flow_judge_mt_bench_prompt,
        process_judge_response=process_judge_response,
        judge_backend="transformers",
    ).compute,
    corpus_level_fn={
        "judge_score_turn_1": np.mean,
        "judge_score_turn_2": np.mean,
    },
)

task = LightevalTaskConfig(
    name="mt_bench",
    prompt_function=mt_bench_prompt,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["extended"],
    hf_repo="lighteval/mt-bench",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="",
    few_shots_select="random",
    metric=[llm_judge_mt_bench],
    generation_size=1024,
    stop_sequence=[],
)

TASKS_TABLE = [task]