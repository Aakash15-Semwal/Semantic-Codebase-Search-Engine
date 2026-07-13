import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from Fine_Tuning.dataset_preparer import DatasetPreparer
from Fine_Tuning.helper import encode_batch, compute_loss, evaluate_model, collate_fn

train_model = AutoModel.from_pretrained("microsoft/codebert-base") 
train_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# 1. DataLoader Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model.to(device)

dataset_preparer = DatasetPreparer()
dataset_preparer.prepare_dataset()
filtered_dataset = dataset_preparer.filtered_dataset

train_loader = DataLoader(
    filtered_dataset['train'],
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# 2. Optimizer Setup
optimizer = AdamW(train_model.parameters(), lr=1e-5)

# Learning rate decay — linear decay to 0 over all training steps, with a short warmup
num_epochs = 1
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 3. Metric tracking — stored every batch, saved periodically so a crash doesn't lose everything
history = {
    "batch_loss": [],
    "batch_lr": [],
    "epoch_avg_loss": []
}

best_mrr = 0

train_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    for batch_idx, (doc_inputs, code_inputs) in enumerate(progress_bar):

        doc_inputs = {k: v.to(device) for k, v in doc_inputs.items()}
        code_inputs = {k: v.to(device) for k, v in code_inputs.items()}

        optimizer.zero_grad()
        text_embeddings, code_embeddings = encode_batch(
            doc_inputs, code_inputs, train_model, train_tokenizer
        )

        loss = compute_loss(text_embeddings, code_embeddings)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        history["batch_loss"].append(loss.item())
        history["batch_lr"].append(scheduler.get_last_lr()[0])

        if batch_idx % 10 == 0:
            progress_bar.set_postfix({'loss': f'{loss.item():.8f}'})
                
        if batch_idx % 500 == 0:
            with open("training_history.json", "w") as f:
                json.dump(history, f)
            torch.save(train_model.state_dict(), f"model{batch_idx}.pt")
            
        if batch_idx > 0 and batch_idx % 1500 == 0:
            metrics = evaluate_model(
                train_model,
                train_tokenizer,
                filtered_dataset["validation"],
                device,
                sample_size=2000
            )
        
            print(metrics)
        
            if metrics["MRR@10"] > best_mrr:
                best_mrr = metrics["MRR@10"]
                torch.save(train_model.state_dict(), "best_model.pt")
        
            train_model.train()

    avg_loss = total_loss / len(train_loader)
    history["epoch_avg_loss"].append(avg_loss)
    print(f'Epoch {epoch+1} Complete. Average Loss: {avg_loss:.8f}')

# Final save at the end of training
with open("training_history.json", "w") as f:
    json.dump(history, f)
    
train_model.save_pretrained("model")
train_tokenizer.save_pretrained("model")