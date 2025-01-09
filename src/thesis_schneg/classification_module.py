from dataclasses import dataclass
from functools import cached_property
from json import load
from pathlib import Path
from typing import Protocol, Mapping, Any, Union

from pandas import DataFrame
from safetensors.torch import load_model
from torch import device, nn, softmax, argmax
from torch.cuda import is_available as cuda_is_available
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    pipeline,
    Pipeline,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,


)
from huggingface_hub import PyTorchModelHubMixin


class _Predictor(Protocol):
    def predict_batch(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.predict_batch(batch)

    @cached_property
    def _device(self) -> device:
        return device("cuda" if cuda_is_available() else "cpu")


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size,
                            len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return softmax(outputs[:, 0, :], dim=1)

# class RegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = AutoModelForSequenceClassification.from_pretrained("AdamCodd/distilroberta-query-wellformedness")
#         self.regression_head = nn.Linear(self.model.config.hidden_size, 1)

#     def forward(self, input_ids, attention_mask, **kwargs):
#         outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask)
#         rating = self.regression_head(outputs.last_hidden_state[:, 0, :])
#         rating = sigmoid(rating)
#         return rating.squeeze()


# @dataclass(frozen=True)
# class WellFormednessClassifier(_Predictor):

#     @cached_property
#     def _tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#         return AutoTokenizer.from_pretrained("AdamCodd/distilroberta-query-wellformedness")

#     @cached_property
#     def _model(self) -> RegressionModel:
#         model = RegressionModel()
#         model = model.regression_head.load_state_dict(torch.load("path_to_the_regression_head.pth"))
#         model = model.to(self._device)
#         model = model.eval()
#         return model

#     def predict_batch(self, batch: DataFrame) -> DataFrame:
#         inputs = self._tokenizer(list(
#             batch["serp_query_text_url"]), return_tensors="pt", padding="longest", truncation=True).to(self._device)
#         outputs = self._model(inputs["input_ids"], inputs["attention_mask"])
#         predicted_classes = argmax(outputs, dim=1)
#         predicted_domains = [self._config.id2label[class_idx.item()]
#                              for class_idx in predicted_classes.cpu().numpy()]
#         batch["query-quality"] = predicted_domains
#         return batch


@dataclass(frozen=True)
class nvidiaQualityClassifier(_Predictor):
    @cached_property
    def _config(self) -> Any:
        return AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")

    @cached_property
    def _tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")

    @cached_property
    def _model(self) -> CustomModel:
        model = CustomModel.from_pretrained(
            "nvidia/quality-classifier-deberta")
        model = model.to(self._device)
        model = model.eval()
        return model

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        inputs = self._tokenizer(list(
            batch["serp_query_text_url"]), return_tensors="pt", padding="longest", truncation=True).to(self._device)
        outputs = self._model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = argmax(outputs, dim=1)
        predicted_domains = [self._config.id2label[class_idx.item()]
                             for class_idx in predicted_classes.cpu().numpy()]
        batch["query-quality"] = predicted_domains
        return batch


@dataclass(frozen=True)
class nvidiaDomainClassifier(_Predictor):
    @cached_property
    def _config(self) -> Any:
        return AutoConfig.from_pretrained("nvidia/domain-classifier")

    @cached_property
    def _tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return AutoTokenizer.from_pretrained("nvidia/domain-classifier")

    @cached_property
    def _model(self) -> CustomModel:
        model = CustomModel.from_pretrained("nvidia/domain-classifier")
        model = model.to(self._device)
        model = model.eval()
        return model

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        inputs = self._tokenizer(list(
            batch["serp_query_text_url"]), return_tensors="pt", padding="longest", truncation=True).to(self._device)
        outputs = self._model(inputs["input_ids"], inputs["attention_mask"])
        predicted_classes = argmax(outputs, dim=1)
        predicted_domains = [self._config.id2label[class_idx.item()]
                             for class_idx in predicted_classes.cpu().numpy()]
        batch["query-domain"] = predicted_domains
        return batch


@dataclass(frozen=True)
class QueryIntentPredictor(_Predictor):
    labels_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/labels.json"
    )
    model_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/bert-orcas-i-level1-query.safetensors"
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
        batch["query-intent"] = [prediction[0]["label"]
                                 for prediction in predictions]
        return batch


@dataclass(frozen=True)
class NSFWPredictor(_Predictor):
    model_name: str = "eliasalbouzidi/distilbert-nsfw-text-classifier"

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
        batch["query-nsfw"] = [prediction[0]["label"]
                               for prediction in predictions]
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
