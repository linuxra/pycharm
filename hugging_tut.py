from transformers import pipeline
import torch
import torch.nn.functional as F

classifier = pipeline("sentiment-analysis")
result = classifier("This world filled with void and free")

print(result)