import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Part 1: Importing Required Libraries

# Part 2: Loading the Pretrained RoBERTa Model
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Part 3: Text Preprocessing and Tokenization
def preprocess_text(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    return input_ids

def predict(text):
    input_ids = preprocess_text(text)
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
    return last_hidden_states

# Part 4: Making Predictions
def detect_ai(text):
    try:
        last_hidden_states = predict(text)
        print("Input tensor shape:", last_hidden_states.shape)
        logits = model.classifier(last_hidden_states)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_labels = probs.topk(1)
        label = top_labels.item()
        return label
    except Exception as e:
        print("An error occurred during prediction: ", e)
        return -1

# Example Usage
sample_text = "This code is just a general structure, and without testing it with a specific use case or dataset, it is not possible to determine if there are any errors in it. However, as long as you have installed the required libraries"

result = detect_ai(sample_text)
if result == -1:
    print("Error occurred during prediction")
else:
    print("Detected AI Label: ", result)

