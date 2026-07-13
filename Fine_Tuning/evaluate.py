from transformers import AutoModel
from torch.utils.data import DataLoader
from Fine_Tuning.helper import evaluate_model
from Fine_Tuning.dataset_preparer import DatasetPreparer

dataset_preparer = DatasetPreparer()
dataset_preparer.prepare_dataset()
filtered_dataset = dataset_preparer.filtered_dataset

train_model = AutoModel.from_pretrained(r".\model").to(device)
base_model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_SIZE = None # Evaluates on the full test set

    print("Evaluating zero-shot CodeBERT...")
    zs_results = evaluate_model(base_model, filtered_dataset['test'], device, sample_size=SAMPLE_SIZE)
    for metric, val in zs_results.items():
        print(f"{metric}: {val:.8f}")

    print("\nEvaluating fine-tuned CodeBERT...")
    ft_results = evaluate_model(train_model, filtered_dataset['test'], device, sample_size=SAMPLE_SIZE)
    for metric, val in ft_results.items():
        print(f"{metric}: {val:.8f}")