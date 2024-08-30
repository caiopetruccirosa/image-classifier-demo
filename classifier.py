from transformers import AutoImageProcessor, AutoModelForImageClassification

import torch
import torch.nn.functional as F


class ImageClassifier:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.model = AutoModelForImageClassification.from_pretrained(self.model_checkpoint)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_checkpoint)

    def predict(self, image):
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0] # remove the batch dimension
        probs = F.softmax(logits, dim=-1)

        scores = []
        for i, prob in enumerate(probs):
            scores.append({
                "score": prob.item(),
                "label": self.model.config.id2label[i]
            })
            
        return scores