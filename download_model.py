from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("Aakash1001/Semantic-CodeBERT")
tokenizer = AutoTokenizer.from_pretrained("Aakash1001/Semantic-CodeBERT")

model.save_pretrained("model")
tokenizer.save_pretrained("model")