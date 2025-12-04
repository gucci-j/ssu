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


def lang_code_to_2en_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to translate to English.

    Args:
        lang_code: The language code 

    Returns:
        The translation instruction string.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "am":
        return "አማርኛን ወደ እንግሊዝኛ ተርጉም:\n"
    elif lang_code == "ne":
        return "नेपालीलाई अङ्ग्रेजीमा अनुवाद गर्नुहोस्:\n"
    elif lang_code == "ha":
        return "Fassara Hausa zuwa Turanci:\n"
    elif lang_code == "ig":
        return "Sụgharịa Igbo gaa na Bekee:\n"
    elif lang_code == "ky":
        return "Кыргызчадан англисчеге которуу:\n"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def lang_code_to_2tgt_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to translate from English to the target language.

    Args:
        lang_code: The language code

    Returns:
        The translation instruction string.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "am":
        return "እንግሊዝኛን ወደ አማርኛ ተርጉም:\n"
    elif lang_code == "ne":
        return "अङ्ग्रेजीलाई नेपालीमा अनुवाद गर्नुहोस्:\n"
    elif lang_code == "ha":
        return "Fassara Turanci zuwa Hausa:\n"
    elif lang_code == "ig":
        return "Sụgharịa Bekee gaa n'Igbo:\n"
    elif lang_code == "ky":
        return "Англисчеден кыргызчага которуу:\n"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn_2en(
    language: str, 
    instruction: str,
):
    def prompt_fn(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}{line[language]} =",
            gold_index=0,
            choices=[line["en"]],
            instruction=instruction,
        )
    return prompt_fn


def buffer_fn_2tgt(
    language: str,
    instruction: str,
):
    def prompt_fn(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}{line['en']} =",
            gold_index=0,
            choices=[line[language]],
            instruction=instruction,
        )
    return prompt_fn


for language in [
    "am",  # Amharic
    "ne",  # Nepali
    "ha",  # Hausa
    "ig",  # Igbo
    "ky",  # Kyrgyz
]:
    task = LightevalTaskConfig(
        name=f"mt:{language}2en",
        prompt_function=buffer_fn_2en(
            language=language,
            instruction=lang_code_to_2en_instruction(language),
        ),
        suite=("custom",),
        hf_repo="your-hf-id/flores-ssu",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)

    task = LightevalTaskConfig(
        name=f"mt:en2{language}",
        prompt_function=buffer_fn_2tgt(
            language=language,
            instruction=lang_code_to_2tgt_instruction(language),
        ),
        suite=("custom",),
        hf_repo="your-hf-id/flores-ssu",
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)
