import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def collate_fn(batch):
    # Extract lists of documentation and code strings
    docstrings = [item['func_documentation_string'] for item in batch]
    codes = [item['func_code_string'] for item in batch]
    
    # Tokenize docstrings
    doc_inputs = tokenizer(
        docstrings, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    
    # Tokenize code snippets
    code_inputs = tokenizer(
        codes, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    
    return doc_inputs, code_inputs

def compute_loss(text_embeddings, code_embeddings, temperature=0.07):
    """
    Computes symmetric InfoNCE loss for a batch of text and code embeddings.
    """
    # Since embeddings are already L2-normalized, matrix multiplication is cosine similarity
    # logits shape: (batch_size, batch_size)
    logits = torch.matmul(text_embeddings, code_embeddings.t()) / temperature

    # The ground truth labels are the diagonal indices (where text i matches code i)
    batch_size = text_embeddings.size(0)
    labels = torch.arange(batch_size, device=text_embeddings.device)

    # Symmetric Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    
    loss_text_to_code = loss_func(logits, labels)
    loss_code_to_text = loss_func(logits.t(), labels)

    return (loss_text_to_code + loss_code_to_text) / 2

def mean_pool(last_hidden_state, attention_mask, input_ids, tokenizer):
    """Same pooling logic as training/production — [CLS]/[SEP]/[PAD] excluded."""
    attention_mask = attention_mask.clone()
    attention_mask[:, 0] = 0
    sep_positions = (input_ids == tokenizer.sep_token_id).int().argmax(dim=1)
    attention_mask[torch.arange(attention_mask.shape[0], device=attention_mask.device), sep_positions] = 0

    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

def encode_batch(doc_inputs, code_inputs, train_model, train_tokenizer):
    """Run both batches through the model, mean-pool, return normalized embeddings."""
    doc_outputs = train_model(**doc_inputs)
    code_outputs = train_model(**code_inputs)

    text_embeddings = mean_pool(
        doc_outputs.last_hidden_state, doc_inputs['attention_mask'], doc_inputs['input_ids'], train_tokenizer
    )
    code_embeddings = mean_pool(
        code_outputs.last_hidden_state, code_inputs['attention_mask'], code_inputs['input_ids'], train_tokenizer
    )

    return text_embeddings, code_embeddings

def encode_texts(texts, model, tokenizer, device, batch_size=32):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs['attention_mask'], inputs['input_ids'], tokenizer)
            all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


def compute_metrics(text_embeddings, code_embeddings, ks=[1, 5, 10]):
    similarity = torch.matmul(text_embeddings, code_embeddings.t())
    n = similarity.size(0)
    
    mrr_sum = 0
    recall_counts = {k: 0 for k in ks}
    
    for i in range(n):
        ranked_indices = torch.argsort(similarity[i], descending=True)
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item() + 1
        
        # MRR@10
        if rank <= 10:
            mrr_sum += 1.0 / rank
            
        # Recall@k
        for k in ks:
            if rank <= k:
                recall_counts[k] += 1
                
    results = {"MRR@10": mrr_sum / n}
    for k in ks:
        results[f"Recall@{k}"] = recall_counts[k] / n
        
    return results

def evaluate_model(model, tokenizer, test_dataset, device, sample_size=None):
    data = test_dataset
    if sample_size is not None and sample_size < len(data):
        data = data.shuffle(seed=42).select(range(sample_size))

    docstrings = data['func_documentation_string']
    codes = data['func_code_string']

    text_embeddings = encode_texts(docstrings, model, tokenizer, device)
    code_embeddings = encode_texts(codes, model, tokenizer, device)

    return compute_metrics(text_embeddings, code_embeddings)
