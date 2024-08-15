from typing import Dict
import numpy as np
from transformers import pipeline

import ray.data

# Step 1: Create a Ray Dataset from in-memory Numpy arrays.
# You can also create a Ray Dataset from many other sources and file
# formats.
ds = ray.data.from_numpy(np.asarray(
    ["Complete this", "for me", "Complete this", "for me"]))

# Step 2: Define a Predictor class for inference.
# Use a class to initialize the model just once in `__init__`
# and re-use it for inference across multiple batches.


class LanguagePredictor:
    def __init__(self):
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.model = pipeline("text-classification", model=model_ckpt)

    def __call__(self, batch: Dict[str, str]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["data"]), top_k=1, truncation=True)

        batch["output"] = [sequences[0]['label']
                           for sequences in predictions]

        return batch


class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline
        # Initialize a pre-trained GPT2 Huggingface pipeline.
        self.model = pipeline("text-generation", model="gpt2")

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Get the predictions from the input batch.
        predictions = self.model(
            list(batch["data"]), max_length=20, num_return_sequences=1)
        # `predictions` is a list of length-one lists. For example:
        # [[{'generated_text': 'output_1'}], ..., [{'generated_text': 'output_2'}]]
        # Modify the output to get it into the following format instead:
        # ['output_1', 'output_2']
        batch["output"] = [sequences[0]["generated_text"]
                           for sequences in predictions]
        return batch


# Step 2: Map the Predictor over the Dataset to get predictions.
# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
predictions = ds.map_batches(LanguagePredictor, concurrency=2)
# Step 3: Show one prediction output.
predictions.show(limit=4)
