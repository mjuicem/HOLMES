import random
import time
import csv
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')
import tiktoken

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model(lm_config, model_path):
    tokenizer = AutoTokenizer.from_pretrained('./model/HOLMES_tokenizer')
    model_from = 1
    if model_from == 1:
        model = Transformer(lm_config)
        state_dict = torch.load(model_path, map_location=device)
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        for k in list(state_dict.keys()):
            if 'mask' in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B')
    return model, tokenizer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_qa_pairs(csv_path, start_idx, end_idx):
    qa_pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if start_idx <= i < end_idx:
                qa_pairs.append({
                    'question': row['q'],
                    'expected_answer': row['a']
                })
    return qa_pairs

class MetricsTracker:
    def __init__(self):
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

def evaluate_response(model_response, expected_answer, metrics):
    model_response = model_response.strip().lower()
    expected_answer = expected_answer.strip().lower()
    matches = model_response == expected_answer

    if expected_answer == 'true story':
        if model_response == 'true story':
            metrics.true_positive += 1
        else:
            metrics.false_negative += 1
    else:
        if model_response == 'deceptive story':
            metrics.true_negative += 1
        else:
            metrics.false_positive += 1
    return matches

if __name__ == "__main__":
    # Initialize lists to store all predictions across all models
    all_preds, all_labels = [], []
    
    csv_path = './100.csv'
    temperature = 0.7
    top_k = 16
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 1024
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    contain_history_chat = False
    stream = True

    # Process each model
    for model_num in range(1, 6):  # Models 1 through 5
        print(f"\nProcessing Model {model_num}")
        model_path = f'./out/full_sft_deception_{model_num}.pth'
        start_idx = (model_num - 1) * 20
        end_idx = model_num * 20

        metrics = MetricsTracker()
        model, tokenizer = init_model(lm_config, model_path)
        model = model.eval()
        qa_pairs = load_qa_pairs(csv_path, start_idx, end_idx)

        correct_count = 0
        total_count = len(qa_pairs)

        for i, qa_pair in enumerate(qa_pairs):
            random_seed = random.randint(0, 2**32 - 1)
            setup_seed(random_seed)
            prompt = qa_pair['question']
            messages = [{"role": "user", "content": prompt}]

            new_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )[-(max_seq_len - 1):]

            x = tokenizer(new_prompt).data['input_ids']
            x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

            with torch.no_grad():
                full_response = ""
                res_y = model.generate(
                    x, tokenizer.eos_token_id, max_new_tokens=max_seq_len,
                    temperature=temperature, top_k=top_k, stream=stream
                )

                history_idx = 0
                while True:
                    try:
                        y = next(res_y)
                    except StopIteration:
                        break
                    answer = tokenizer.decode(y[0].tolist())
                    if not answer or answer[-1] == '�':
                        continue
                    full_response += answer[history_idx:]
                    history_idx = len(answer)

            matches = evaluate_response(
                full_response.replace(new_prompt, ""), 
                qa_pair['expected_answer'], metrics
            )
            if matches:
                correct_count += 1

            # Print model's answer
            global_index = start_idx + i + 1
            print(f"R[{global_index}]")
            print(f"A[{global_index}]: {full_response.replace(new_prompt, '')}")
            print(f"Ground Truth: {qa_pair['expected_answer']}")

            # For additional metrics
            label = 1 if qa_pair['expected_answer'].lower() == "true story" else 0
            pred = 1 if "true story" in full_response.lower() else 0
            all_labels.append(label)
            all_preds.append(pred)

            # Print real-time accuracy for current model
            current_accuracy = correct_count / (i + 1) * 100
            print(f"Current Model Acc Until R[{global_index}]: {current_accuracy:.2f}%\n")

        # Free up GPU memory
        del model
        torch.cuda.empty_cache()

    # Calculate final metrics using all predictions
    accuracy = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds) * 100

    # Print final aggregated metrics
    print("\nFinal Aggregated Metrics Across All Models:")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-score: {f1:.2f}%")
    print("Confusion Matrix (TN, FP, FN, TP):", cm.ravel())
    print(f"ROC AUC: {roc_auc:.2f}%")