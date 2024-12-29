from safetensors import safe_open

tensors = {}
with safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/bert-orcas-i-level1-query.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)  # loads the full tensor given a key
print(tensors)
