import torch
from transformers import AutoTokenizer, AutoModel


class CodeEmbedder:
    def __init__(self, model_name=r".\model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _mean_pool(self, last_hidden_state, attention_mask, input_ids):
        """Mean pooling excluding [CLS], [SEP], and [PAD] tokens."""
        attention_mask = attention_mask.clone()
        attention_mask[:, 0] = 0  # exclude [CLS]
        sep_positions = (input_ids == self.tokenizer.sep_token_id).int().argmax(dim=1)
        attention_mask[torch.arange(attention_mask.shape[0]), sep_positions] = 0  # exclude [SEP]

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    def embed_chunks(self, chunks, batch_size=8):
        """Embeds a list of chunk dicts in place, adding an 'embedding' field to each."""
        code_list = [chunk['code'] for chunk in chunks]
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(code_list), batch_size):
                batch_codes = code_list[i: i + batch_size]
                inputs = self.tokenizer(
                    batch_codes, padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                )
                outputs = self.model(**inputs)
                normalized = self._mean_pool(
                    outputs.last_hidden_state, inputs.attention_mask, inputs.input_ids
                )
                all_embeddings.append(normalized)

        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()

        for idx, chunk in enumerate(chunks):
            chunk['embedding'] = all_embeddings[idx]

        return chunks

    def embed_query(self, query):
        """Embeds a single natural language query string."""
        inputs = self.tokenizer(
            query, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            normalized = self._mean_pool(
                outputs.last_hidden_state, inputs.attention_mask, inputs.input_ids
            )
        return normalized.cpu().numpy()