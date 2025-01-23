# import torch
# from torch import nn
# from huggingface_hub import PyTorchModelHubMixin

# from dataclasses import dataclass
# from functools import cached_property
# from typing import Any, Union, Protocol

# from pandas import DataFrame
# from torch import device, argmax
# from torch.cuda import is_available as cuda_is_available
# from transformers import (
#     AutoModel,
#     AutoTokenizer,
#     AutoConfig,
#     PreTrainedTokenizer,
#     PreTrainedTokenizerFast,
#     Pipeline,
#     pipeline,
# )


# class _Predictor(Protocol):
#     def predict_batch(self, batch: DataFrame) -> DataFrame:
#         raise NotImplementedError()

#     def __call__(self, batch: DataFrame) -> DataFrame:
#         return self.predict_batch(batch)

#     @cached_property
#     def _device(self) -> device:
#         return device("cuda" if cuda_is_available() else "cpu")


# class CustomModel(nn.Module, PyTorchModelHubMixin):
#     def __init__(self, config):
#         super(CustomModel, self).__init__()
#         self.model = AutoModel.from_pretrained(config["base_model"])
#         self.dropout = nn.Dropout(config["fc_dropout"])
#         self.fc = nn.Linear(self.model.config.hidden_size,
#                             len(config["id2label"]))

#     def forward(self, input_ids, attention_mask):
#         features = self.model(input_ids=input_ids,
#                               attention_mask=attention_mask).last_hidden_state
#         dropped = self.dropout(features)
#         outputs = self.fc(dropped)
#         return torch.softmax(outputs[:, 0, :], dim=1)


# @dataclass(frozen=True)
# class nvidiaDomainClassifier(_Predictor):
#     @cached_property
#     def _config(self) -> Any:
#         return AutoConfig.from_pretrained("nvidia/domain-classifier")

#     @cached_property
#     def _tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#         return AutoTokenizer.from_pretrained("nvidia/domain-classifier")

#     @cached_property
#     def _model(self) -> CustomModel:
#         model = CustomModel.from_pretrained("nvidia/domain-classifier")
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
#         batch["query-domain"] = predicted_domains
#         return batch


# @dataclass(frozen=True)
# class nvidiaQualityClassifier(_Predictor):
#     @cached_property
#     def _config(self) -> Any:
#         return AutoConfig.from_pretrained("nvidia/quality-classifier-deberta")

#     @cached_property
#     def _tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#         return AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")

#     @cached_property
#     def _model(self) -> CustomModel:
#         model = CustomModel.from_pretrained(
#             "nvidia/quality-classifier-deberta")
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


# @dataclass(frozen=True)
# class NSFWPredictor(_Predictor):
#     model_name: str = "eliasalbouzidi/distilbert-nsfw-text-classifier"

#     @cached_property
#     def _pipeline(self) -> Pipeline:
#         return pipeline(
#             task="text-classification",
#             model=self.model_name,
#             device=self._device,
#             padding=True,
#             truncation=True,
#             top_k=1,
#         )

#     def predict_batch(self, batch: DataFrame) -> DataFrame:
#         # Reset call count of classifier pipeline.
#         self._pipeline.call_count = 0
#         predictions = self._pipeline(list(batch["serp_query_text_url"]))
#         batch["label"] = [prediction[0]["label"] for prediction in predictions]
#         return batch


# # Setup configuration and model
# config = AutoConfig.from_pretrained("nvidia/domain-classifier")
# tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
# model = CustomModel.from_pretrained("nvidia/domain-classifier")
# model.eval()

# # Prepare and process inputs
# text_samples = ["Sports is a popular domain", "Politics is a popular domain", "ahdsa skjfh a",
#                 "Today is a good day", "This is a test sample. If this works, then it is a success."]
# inputs = tokenizer(text_samples, return_tensors="pt",
#                    padding="longest", truncation=True)
# outputs = model(inputs["input_ids"], inputs["attention_mask"])

# # Predict and display results
# predicted_classes = torch.argmax(outputs, dim=1)
# predicted_domains = [config.id2label[class_idx.item()]
#                      for class_idx in predicted_classes.cpu().numpy()]
# print(predicted_domains)
# # ['Sports', 'News']


# print("####################################")


# pred = nvidiaDomainClassifier()

# print(pred.predict_batch(DataFrame({"serp_query_text_url": text_samples})))
# print(pred(DataFrame({"serp_query_text_url": text_samples})))

# print("####################################")

# pred = nvidiaQualityClassifier()

# print(pred.predict_batch(DataFrame({"serp_query_text_url": text_samples})))
# print(pred(DataFrame({"serp_query_text_url": text_samples})))

# print("####################################")
# text_samples = ["fuck you", "Hello, how are you?"]
# pred = NSFWPredictor()

# print(pred.predict_batch(DataFrame({"serp_query_text_url": text_samples})))
# print(pred(DataFrame({"serp_query_text_url": text_samples})))


from gliner import GLiNER
import time
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

text = """
Harilala Rasoanaivo, un homme d'affaires local d'Antananarivo, a enregistré une nouvelle société nommée "Rasoanaivo Enterprises" au Lot II M 92 Antohomadinika. Son numéro est le +261 32 22 345 67, et son adresse électronique est harilala.rasoanaivo@telma.mg. Il a fourni son numéro de sécu 501-02-1234 pour l'enregistrement."""
textarray = []
for i in range(1024):
    textarray.append(text)

labels = ["person", "organization", "phone number", "address", "passport number", "email", "credit card number", "social security number", "health insurance id number", "date of birth", "mobile phone number", "bank account number", "medication", "cpf", "driver's license number", "tax identification number", "medical condition", "identity card number", "national id number", "ip address", "email address", "iban", "credit card expiration date", "username", "health insurance number", "registration number", "student id number",
          "insurance number", "flight number", "landline phone number", "blood type", "cvv", "reservation number", "digital signature", "social media handle", "license plate number", "cnpj", "postal code", "passport_number", "serial number", "vehicle registration number", "credit card brand", "fax number", "visa number", "insurance company", "identity document number", "transaction number", "national health insurance number", "cvc", "birth certificate number", "train ticket number", "passport expiration date", "social_security_number"]
entity_text = []
entitiy_label = []
start = time.time()
for text in textarray:
    entities = model.predict_entities(text, labels)
    for entity in entities:
        entity_text.append(entity["text"])
        entitiy_label.append(entity["label"])
end = time.time()
print("Time taken:", end-start)


# text = "ajdgg JHKDFB middestreet 5 london shj akld paris hjab a,mklö .-_.; ()Z$ P§ OI()?R peter fox"


# labels = ["person", "organization", "phone number", "address", "passport number", "email", "credit card number", "social security number", "health insurance id number", "date of birth", "mobile phone number", "bank account number", "medication", "cpf", "driver's license number", "tax identification number", "medical condition", "identity card number", "national id number", "ip address", "email address", "iban", "credit card expiration date", "username", "health insurance number", "registration number", "student id number",
#           "insurance number", "flight number", "landline phone number", "blood type", "cvv", "reservation number", "digital signature", "social media handle", "license plate number", "cnpj", "postal code", "passport_number", "serial number", "vehicle registration number", "credit card brand", "fax number", "visa number", "insurance company", "identity document number", "transaction number", "national health insurance number", "cvc", "birth certificate number", "train ticket number", "passport expiration date", "social_security_number"]
# start = time.time()
# entities = model.predict_entities(text, labels)
# for entity in entities:
#     print(entity["text"], "=>", entity["label"])
# end = time.time()
# print("Time taken:", end-start)
