from transformers import pipeline

classfier = pipeline('sentiment-analysis')
classfier("I've been waiting for a HuggingFace course my whole life.")