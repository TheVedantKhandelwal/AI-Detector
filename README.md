
# AI-Detect

Python code to use the pre-trained RoBERTa Language Model, take Text Input and predict the use of AI

## Requirements
- PyTorch
- transformers
- Patience

    
## Usage/Examples
1. Load the  pretrained RoBERTa Model:
 ```python
model = RobertaForSequenceClassification.from_pretrained('roberta-base')
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```
2. Pre-Process the text input by tokenizing it
```python
def preprocess_text(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    return input_ids
```
3. Make predictions using the model:
```python
def detect_ai(text):
    try:
        last_hidden_states = predict(text)
        logits = model.classifier(last_hidden_states)
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_labels = probs.topk(1)
        label = top_labels.item()
        return label
    except Exception as e:
        print("An error occurred during prediction: ", e)
        return -1
```
### Example Usage : 
```python
sample_text = "The importance of CGPA (Cumulative Grade Point Average) depends on several factors, such as the type of university, academic program, and future career goals. In general, CGPA is often considered an important factor in the admission process for graduate programs, and it is also used by some employers as a way to assess the academic performance of job candidates."

result = detect_ai(sample_text)
if result == -1:
    print("Error occurred during prediction")
else:
    print("Detected AI Label: ", result) 
```

