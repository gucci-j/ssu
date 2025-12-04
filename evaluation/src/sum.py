from aenum import extend_enum
import numpy as np

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TASKS_TABLE = []

# CUSTOM METRIC IF NEEDED
class SampleLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        import sacrebleu
        self.metric_type = metric_type
        if metric_type == "chrf":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "chrf++":
            self.metric = sacrebleu.sentence_chrf
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        assert len(golds) == 1 and len(predictions) == 1
        if self.metric_type == "chrf++":
            return float(self.metric(predictions.pop(), golds, word_order=2).score)
        else:
            return float(self.metric(predictions.pop(), golds).score)

chrf_sample = SampleLevelMetric(
    metric_name="chrfpp_sample",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    sample_level_fn=SampleLevelTranslationMetric("chrf++").compute, # how to compute score for one sample
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)
extend_enum(Metrics, "chrfpp_sample", chrf_sample)


def lang_code_to_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The instruction in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summarize the following text in English:"
    elif lang_code == "am":
        return "የታችኛው ጽሁፍን በአማርኛ አጭር በማድረግ አሳትረኝ።:"
    elif lang_code == "ne":
        return "तलको पाठलाई नेपालीमा संक्षेपमा लेख्नुहोस्:"
    elif lang_code == "ha":
        return "Taƙaita rubutu mai zuwa cikin Hausa:"
    elif lang_code == "ig":
        return "Chịkọta edemede a n'Igbo:"
    elif lang_code == "ky":
        return "Төмөнкү текстти кыргызча кыскача жазыңыз:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def lang_code_to_anchor(lang_code: str) -> str:
    """Converts a language code to an anchor to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The anchor in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summary:"
    elif lang_code == "am":
        return "አጭር መግለጫ:"
    elif lang_code == "ne":
        return "सारांश:"
    elif lang_code == "ha":
        return "Taƙaitawa:"
    elif lang_code == "ig":
        return "Nchịkọta:"
    elif lang_code == "ky":
        return "Кыскача:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn(
    instruction: str,
    anchor: str,
):
    def prompt_fn(line, task_name: str):
        summary = line["summary"]
        text = line["text"]
        return Doc(
            task_name=task_name,
            query=f"{instruction} {text} {anchor}",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text}
        )
    return prompt_fn


for language in [
    "en",
    "am",
    "ne",
    "ha",
    "ig",
    "ky",
]:
    task = LightevalTaskConfig(
        name=f"sum:{language}",
        prompt_function=buffer_fn(
            instruction=lang_code_to_instruction(language),
            anchor=lang_code_to_anchor(language),
        ),
        suite=("custom",),
        hf_repo=f"your-hf-id/sum-{language}-ssu",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)
