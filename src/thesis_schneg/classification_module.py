from dataclasses import dataclass
from functools import cached_property
from json import load
from pathlib import Path
from typing import Protocol, Iterable, Mapping, Optional

from pandas import DataFrame
from safetensors.torch import load_model
from torch import device
from torch.cuda import is_available as cuda_is_available
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    pipeline,
    Pipeline,
)


class _Predictor(Protocol):
    def predict_batch(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.predict_batch(batch)

    @cached_property
    def _device(self) -> device:
        return device("cuda" if cuda_is_available() else "cpu")


@dataclass(frozen=True)
class QueryIntentPredictor(_Predictor):
    labels_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/labels.json"
    )
    model_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/finetuned_BERT_first_level.safetensors"
    )

    @cached_property
    def _id2label(self) -> Mapping[int, str]:
        # Load labels.
        with self.labels_path.open("rb") as file:
            label_dict = load(fp=file)

        # Swap label dict keys/values.
        return {v: k for k, v in label_dict.items()}

    @cached_property
    def _tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            do_lower_case=True,
        )

    @cached_property
    def _model(self) -> BertForSequenceClassification:
        # Load the BERT model.
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            id2label=self._id2label,
            output_attentions=False,
            output_hidden_states=False,
        )

        model.to(self._device)

        # Load state of trained model.
        load_model(
            model=model,
            filename=str(self.model_path),
            strict=False,
            device=self._device.type,
        )

        return model

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        # Reset call count of classifier pipeline.
        self._pipeline.call_count = 0
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["label"] = [prediction[0]["label"] for prediction in predictions]
        return batch


@dataclass(frozen=True)
class LanguagePredictor(_Predictor):
    model_name: str = "papluca/xlm-roberta-base-language-detection"

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self.model_name,
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        # Reset call count of classifier pipeline.
        self._pipeline.call_count = 0
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["label"] = [prediction[0]["label"] for prediction in predictions]
        return batch


@dataclass(frozen=True)
class HateSpeechPredictor(_Predictor):
    model_name: str = "facebook/roberta-hate-speech-dynabench-r4-target"

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self.model_name,
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["label"] = [prediction[0]["label"] for prediction in predictions]
        return batch


@dataclass(frozen=True)
class SpamPredictor(_Predictor):
    model_name: str = "mshenoda/roberta-spam"

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self.model_name,
            model_kwargs=dict(
                id2label={
                    0: "No Spam",
                    1: "Spam",
                }
            ),
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["label"] = [prediction[0]["label"] for prediction in predictions]
        return batch


@dataclass(frozen=True)
class QueryRatingPredictor(_Predictor):
    """
    Rate the well-formedness of a query in grammatical terms.
    The output rating is a float between 0 and 1.
    """

    model_name: str = "Ashishkr/query_wellformedness_score"

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self.model_name,
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["score"] = [prediction[0]["score"] for prediction in predictions]
        return batch
