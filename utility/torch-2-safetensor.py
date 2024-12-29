import torch
from safetensors.torch import save_file

tensors = torch.load(
    "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/bert-orcas-i-level1-query.model")


save_file(tensors, "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/bert-orcas-i-level1-query.safetensors")
