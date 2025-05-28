import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
# Model Configuration
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Or your specific fine-tuned qwen2-7b path
NUM_CLASSES = 2  # Human (0) vs LLM (1)
NUM_ATTRIBUTION_LABELS = 5 # As per paper

# LoRA Configuration
USE_LORA = True
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32 # LoRA alpha
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Configuration
LEARNING_RATE = 5e-5
EPOCHS = 3  # Keep epochs low for demonstration with dummy data
BATCH_SIZE = 2 # Adjusted to be smaller for dummy data style alignment to work
MAX_LENGTH = 512

# Data Splitting
TEST_SPLIT_RATIO = 0.25 # Adjusted for small dummy data
RANDOM_STATE_SPLIT = 42

# Loss Weights
ALPHA_STYLE = 1.0
ALPHA_CLASS = 1.0
ALPHA_ATTR = 1.0

ATTRIBUTION_CATEGORIES = [
    "Single language style", "Too structured logical structure",
    "Lack of background knowledge and personal experience",
    "Repetitive or patterned expression", "Data biases and errors"
]

# Paths
MODEL_SAVE_PATH = "safd_model_final"
METRICS_SAVE_PATH = "evaluation_metrics.json"
PLOTS_SAVE_PATH = "evaluation_plots"
DUMMY_DATA_PATH = "data/human_vs_llm_dataset.jsonl"
TRAIN_SPLIT_PATH = "train_split.jsonl"
TEST_SPLIT_PATH = "test_split.jsonl"

# --- Dataset Class and Collate Function ---
class SAFDDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length): # Modified to take list of dicts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data_list # Use the provided list of dictionaries
        self.styled_human_groups = defaultdict(list)

        for i, entry in enumerate(self.data):
            if entry.get('is_styled_human_variant', False) and 'original_human_id' in entry:
                self.styled_human_groups[entry['original_human_id']].append(i)

    @classmethod
    def from_jsonl(cls, json_path, tokenizer, max_length):
        data_list = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
        return cls(data_list, tokenizer, max_length)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['text']
        label = int(entry['label'])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'class_label': torch.tensor(label, dtype=torch.long),
            'has_attribution': False,
            'is_llm_generated': (label == 1),
            'is_styled_human_variant': entry.get('is_styled_human_variant', False),
            'original_human_id': entry.get('original_human_id', None),
            'idx_in_dataset': idx
        }

        if label == 1 and 'attribution_labels' in entry:
            attr_labels = entry['attribution_labels']
            if isinstance(attr_labels, list) and len(attr_labels) == NUM_ATTRIBUTION_LABELS:
                 item['attribution_labels'] = torch.tensor(attr_labels, dtype=torch.float)
                 item['has_attribution'] = True
            else:
                item['attribution_labels'] = torch.zeros(NUM_ATTRIBUTION_LABELS, dtype=torch.float)
                item['has_attribution'] = False
        else:
            item['attribution_labels'] = torch.zeros(NUM_ATTRIBUTION_LABELS, dtype=torch.float)

        return item

def safd_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    class_labels = torch.stack([item['class_label'] for item in batch])
    attribution_labels = torch.stack([item['attribution_labels'] for item in batch])
    has_attribution_mask = torch.tensor([item['has_attribution'] for item in batch], dtype=torch.bool)
    is_llm_generated_mask = torch.tensor([item['is_llm_generated'] for item in batch], dtype=torch.bool)

    batch_styled_human_groups = defaultdict(list)
    for i, item in enumerate(batch):
        if item['is_styled_human_variant'] and item['original_human_id'] is not None:
            batch_styled_human_groups[item['original_human_id']].append(i)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'class_labels': class_labels,
        'attribution_labels': attribution_labels,
        'has_attribution_mask': has_attribution_mask,
        'is_llm_generated_mask': is_llm_generated_mask,
        'batch_styled_human_groups': batch_styled_human_groups
    }

# --- Model Class ---
class SAFDModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, num_attribution_labels=NUM_ATTRIBUTION_LABELS):
        super(SAFDModel, self).__init__()
        
        base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if USE_LORA:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET_MODULES,
            )
            self.backbone = get_peft_model(base_model, peft_config)
            print("Applied LoRA to the backbone model.")
            self.backbone.print_trainable_parameters()
        else:
            self.backbone = base_model
        
        if hasattr(self.backbone, 'config'):
            hidden_size = self.backbone.config.hidden_size
        else:
            hidden_size = base_model.config.hidden_size

        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.attribution_head = nn.Linear(hidden_size, num_attribution_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, -1, :] 

        class_logits = self.classification_head(pooled_output)
        attr_logits = self.attribution_head(pooled_output)   
        
        return class_logits, attr_logits

    def save_lora_model(self, path):
        if USE_LORA:
            self.backbone.save_pretrained(path)
            print(f"LoRA adapter model saved to {path}")
        else:
            print("LoRA is not used.")

# --- Loss Functions ---
def classification_loss_fn(class_logits, class_labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(class_logits, class_labels)

def attribution_loss_fn(attr_logits, attr_labels, valid_attr_samples_mask):
    if not torch.any(valid_attr_samples_mask):
        return torch.tensor(0.0, device=attr_logits.device)

    filtered_attr_logits = attr_logits[valid_attr_samples_mask]
    filtered_attr_labels = attr_labels[valid_attr_samples_mask]
    
    loss_fct = torch.nn.BCEWithLogitsLoss()
    return loss_fct(filtered_attr_logits, filtered_attr_labels.float())

def style_alignment_loss_fn(class_logits_groups):
    total_kl_loss = 0.0
    num_groups_processed = 0

    for group_logits in class_logits_groups:
        if group_logits.size(0) < 2:
            continue
        
        group_probs = F.softmax(group_logits, dim=-1)
        n_s = group_probs.size(0)
        group_kl_sum = 0.0
        
        for i in range(n_s):
            for j in range(n_s):
                if i == j:
                    continue
                kl_div = F.kl_div(group_probs[j].log(), group_probs[i], reduction='sum', log_target=False) # Added log_target=False
                group_kl_sum += kl_div
        
        if n_s > 0:
             total_kl_loss += group_kl_sum / (n_s * n_s) 
             num_groups_processed +=1

    if not class_logits_groups or num_groups_processed == 0: # Check if list is empty or no groups were processed
        return torch.tensor(0.0, device='cpu' if not class_logits_groups or not class_logits_groups[0].is_cuda else class_logits_groups[0].device)


    return total_kl_loss / num_groups_processed

# --- Utility Functions ---
def load_and_split_data(json_path, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE_SPLIT):
    all_data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))
    
    if not all_data:
        raise ValueError("No data loaded from the JSON file.")

    try:
        labels = [item['label'] for item in all_data]
        if len(set(labels)) > 1:
             train_data, test_data = train_test_split(
                all_data, test_size=test_size, random_state=random_state, stratify=labels
            )
        else:
            print("Warning: Not enough classes for stratified split. Using simple random split.")
            train_data, test_data = train_test_split(
                all_data, test_size=test_size, random_state=random_state
            )
    except KeyError:
        print("Warning: 'label' key not found. Using simple random split.")
        train_data, test_data = train_test_split(
            all_data, test_size=test_size, random_state=random_state
        )
    
    print(f"Data loaded: {len(all_data)} items. Split into {len(train_data)} train and {len(test_data)} test items.")
    return train_data, test_data

def save_data_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def calculate_metrics(y_true, y_pred_proba, y_pred_labels):
    accuracy = accuracy_score(y_true, y_pred_labels)
    # Use zero_division=0 for precision, recall, f1 to handle cases with no true/predicted positives
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='binary', zero_division=0)

    auc_roc = "N/A"
    if len(np.unique(y_true)) > 1:
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
        except ValueError as e:
            print(f"Warning: Could not calculate AUROC: {e}")
            auc_roc = f"N/A ({e})"
    else:
        print("Warning: AUROC cannot be calculated with only one class present in y_true.")

    metrics = {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1_score": f1, "auroc": auc_roc
    }
    return metrics

def plot_confusion_matrix_util(y_true, y_pred_labels, class_names, save_filename=None):
    if not os.path.exists(PLOTS_SAVE_PATH):
        os.makedirs(PLOTS_SAVE_PATH)
    
    cm = confusion_matrix(y_true, y_pred_labels, labels=np.arange(len(class_names))) # Ensure labels are set if some classes are missing in preds
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    
    if save_filename:
        full_save_path = os.path.join(PLOTS_SAVE_PATH, save_filename)
        plt.savefig(full_save_path)
        print(f"Confusion matrix saved to {full_save_path}")
    plt.show()

def create_dummy_data_if_not_exists(dummy_data_path=DUMMY_DATA_PATH):
    if os.path.exists(dummy_data_path):
        print(f"Dummy data file {dummy_data_path} already exists. Using it.")
        return dummy_data_path

    print(f"Creating dummy data at {dummy_data_path}...")
    dummy_data = []
    for i in range(10): # Increased samples
        hid = f"h{i}"
        dummy_data.append({"id": hid, "text": f"An original tale of a brave mouse {i}.", "label": 0, "is_styled_human_variant": False})
        dummy_data.append({"id": f"{hid}_sA", "text": f"Once upon a time, mouse {i} ventured forth, with whiskers twitching.", "label": 0, "is_styled_human_variant": True, "original_human_id": hid})
        dummy_data.append({"id": f"{hid}_sB", "text": f"Subject Mouse {i}: Log entry. Explored cheese mountain. Found crumbs.", "label": 0, "is_styled_human_variant": True, "original_human_id": hid})
    
    for i in range(10):
        attr = [1,1,0,0,0] if i % 3 == 0 else ([0,1,0,1,0] if i % 3 == 1 else [0,0,1,1,1])
        dummy_data.append({"id": f"l{i}", "text": f"The LLM analyzed data point {i}, concluding statistical significance.", "label": 1, "attribution_labels": attr})
        dummy_data.append({"id": f"l_alt{i}", "text": f"Generated narrative {i}: a robot dreamed of electric sheep, a common trope.", "label": 1, "attribution_labels": attr})

    with open(dummy_data_path, 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n')
    print(f"Dummy data created at {dummy_data_path} with {len(dummy_data)} samples.")
    return dummy_data_path

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss_epoch, total_l_class_epoch, total_l_attr_epoch, total_l_style_epoch = 0,0,0,0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        class_labels = batch['class_labels'].to(device)
        attr_labels = batch['attribution_labels'].to(device)
        has_attr_mask = batch['has_attribution_mask'].to(device)
        batch_styled_groups_indices = batch['batch_styled_human_groups']

        class_logits, attr_logits = model(input_ids, attention_mask)

        l_class = classification_loss_fn(class_logits, class_labels)
        l_attr = attribution_loss_fn(attr_logits, attr_labels, has_attr_mask)
        
        style_loss_logit_groups = []
        if batch_styled_groups_indices:
            for _, batch_indices_for_group in batch_styled_groups_indices.items():
                if len(batch_indices_for_group) >= 2: # Ensure at least two items for comparison
                    logits_for_this_group = class_logits[torch.tensor(batch_indices_for_group, device=device)]
                    style_loss_logit_groups.append(logits_for_this_group)
        
        l_style = style_alignment_loss_fn(style_loss_logit_groups) if style_loss_logit_groups else torch.tensor(0.0, device=device)

        current_total_loss = (ALPHA_CLASS * l_class + ALPHA_ATTR * l_attr + ALPHA_STYLE * l_style)

        optimizer.zero_grad()
        current_total_loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.get_last_lr()[0] > 0: # Check if scheduler is active
             scheduler.step()


        total_loss_epoch += current_total_loss.item()
        total_l_class_epoch += l_class.item()
        total_l_attr_epoch += l_attr.item()
        total_l_style_epoch += l_style.item()

        if (batch_idx + 1) % 1 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss: {current_total_loss.item():.4f} (Cls: {l_class.item():.4f}, Attr: {l_attr.item():.4f}, Sty: {l_style.item():.4f})")
    
    num_batches = len(dataloader)
    avg_loss = total_loss_epoch / num_batches if num_batches > 0 else 0
    avg_l_class = total_l_class_epoch / num_batches if num_batches > 0 else 0
    avg_l_attr = total_l_attr_epoch / num_batches if num_batches > 0 else 0
    avg_l_style = total_l_style_epoch / num_batches if num_batches > 0 else 0
    return avg_loss, avg_l_class, avg_l_attr, avg_l_style

def evaluate_model(model, dataloader, device):
    model.eval()
    all_class_labels, all_class_pred_probs, all_class_pred_labels = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            class_labels = batch['class_labels'].to(device)

            class_logits, _ = model(input_ids, attention_mask)
            class_probs = torch.softmax(class_logits, dim=-1)
            _, predicted_labels = torch.max(class_probs, dim=-1)

            all_class_labels.extend(class_labels.cpu().numpy())
            all_class_pred_probs.extend(class_probs.cpu().numpy())
            all_class_pred_labels.extend(predicted_labels.cpu().numpy())
            
            if (batch_idx + 1) % (len(dataloader)//2 +1) == 0 : # Log progress
                 print(f"  Evaluated Batch [{batch_idx+1}/{len(dataloader)}]")


    all_class_labels_np = np.array(all_class_labels)
    all_class_pred_probs_np = np.array(all_class_pred_probs)
    all_class_pred_labels_np = np.array(all_class_pred_labels)
    
    if len(all_class_labels_np) == 0:
        print("Warning: No samples in evaluation set to calculate metrics.")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "auroc": "N/A"}

    metrics = calculate_metrics(all_class_labels_np, all_class_pred_probs_np, all_class_pred_labels_np)
    plot_confusion_matrix_util(all_class_labels_np, all_class_pred_labels_np, class_names=["Human", "LLM"], save_filename="confusion_matrix_test.png")
    
    return metrics

# --- Main Execution Block ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_data_path = create_dummy_data_if_not_exists(DUMMY_DATA_PATH)
    train_data_list, test_data_list = load_and_split_data(raw_data_path)

    save_data_to_jsonl(train_data_list, TRAIN_SPLIT_PATH)
    save_data_to_jsonl(test_data_list, TEST_SPLIT_PATH)
    print(f"Train data saved to {TRAIN_SPLIT_PATH}, Test data saved to {TEST_SPLIT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token_id})")

    # Pass the list of dicts directly
    train_dataset = SAFDDataset(data_list=train_data_list, tokenizer=tokenizer, max_length=MAX_LENGTH)
    test_dataset = SAFDDataset(data_list=test_data_list, tokenizer=tokenizer, max_length=MAX_LENGTH)
    
    train_batch_size = min(BATCH_SIZE, len(train_dataset)) if len(train_dataset) > 0 else 0
    test_batch_size = min(BATCH_SIZE, len(test_dataset)) if len(test_dataset) > 0 else 0

    if len(train_dataset) == 0 or (EPOCHS > 0 and train_batch_size == 0) :
        print("Error: Train dataset is empty or batch size is zero. Cannot train.")
        if len(test_dataset) == 0 or test_batch_size == 0:
            print("Test dataset is also empty or batch size is zero. Exiting.")
            return # Exit if no training and no testing is possible

    if EPOCHS > 0 and train_batch_size > 0:
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=safd_collate_fn)
        print(f"Train Dataloader: {len(train_dataloader)} batches of size {train_batch_size}")
    else:
        train_dataloader = None # No training if epochs is 0 or no data
        print("Skipping training setup as EPOCHS is 0 or train data is insufficient.")

    if len(test_dataset) > 0 and test_batch_size > 0:
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=safd_collate_fn)
        print(f"Test Dataloader: {len(test_dataloader)} batches of size {test_batch_size}")
    else:
        test_dataloader = None
        print("Test dataloader is empty. Evaluation will be skipped.")


    model = SAFDModel().to(device)
    if tokenizer.pad_token_id == tokenizer.eos_token_id and not USE_LORA: # Only if not using LoRA and vocab changed by adding *new* pad
         model.backbone.resize_token_embeddings(len(tokenizer)) # This line can be tricky with PEFT. Usually not needed if pad_token=eos_token.

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    scheduler = None
    if train_dataloader and EPOCHS > 0:
        num_training_steps = len(train_dataloader) * EPOCHS
        if num_training_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps
            )
    
    if train_dataloader and EPOCHS > 0:
        print("\n--- Starting Training ---")
        for epoch in range(EPOCHS):
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            avg_loss, avg_l_class, avg_l_attr, avg_l_style = train_epoch(model, train_dataloader, optimizer, scheduler, device)
            print(f"--- Epoch {epoch+1} Summary ---")
            print(f"  Avg Total Loss: {avg_loss:.4f} (Cls: {avg_l_class:.4f}, Attr: {avg_l_attr:.4f}, Sty: {avg_l_style:.4f})")
            print("-------------------------")
    else:
        print("\n--- Training Skipped ---")


    if test_dataloader:
        print("\n--- Starting Evaluation ---")
        final_metrics = evaluate_model(model, test_dataloader, device)
        print("\n--- Final Evaluation Metrics ---")
        for metric_name, value in final_metrics.items():
            print(f"  {metric_name.capitalize()}: {value if isinstance(value, str) else f'{value:.4f}'}")
        
        with open(METRICS_SAVE_PATH, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Evaluation metrics saved to {METRICS_SAVE_PATH}")
    else:
        print("\n--- Evaluation Skipped ---")


    print("\n--- Saving Model ---")
    save_dir = MODEL_SAVE_PATH
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if USE_LORA:
        model.save_lora_model(os.path.join(save_dir, "lora_adapter"))
    # Always save heads and tokenizer as they are trained/used regardless of LoRA for backbone
    torch.save(model.classification_head.state_dict(), os.path.join(save_dir, "classification_head.pth"))
    torch.save(model.attribution_head.state_dict(), os.path.join(save_dir, "attribution_head.pth"))
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
    print(f"Model components saved to {save_dir}")
    
    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()