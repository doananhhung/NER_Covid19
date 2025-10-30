# src/train.py
#
# Script chính để huấn luyện mô hình NER.
# Script này sẽ:
# 1. Tải cấu hình từ config.py.
# 2. Khởi tạo Dataset và DataLoader.
# 3. Tải mô hình PhoBERT đã được huấn luyện trước.
# 4. Thực hiện vòng lặp huấn luyện và đánh giá.
# 5. Lưu lại checkpoint của mô hình tốt nhất.

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score

# Import các module tự định nghĩa
import config
from dataset import NerDataset

def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """Thực hiện huấn luyện trong một epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Chuyển batch dữ liệu sang device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Xóa các gradient cũ
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass và cập nhật trọng số
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, ids_to_tags):
    """Đánh giá mô hình trên tập validation."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Lấy các dự đoán (logits) và chuyển sang nhãn
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            # Chuyển đổi ID sang Tag và xử lý subword
            for i in range(len(true_labels)):
                pred_tags = [ids_to_tags[p] for p, l in zip(predictions[i], true_labels[i]) if l != config.SUBWORD_TAG_ID]
                label_tags = [ids_to_tags[l] for l in true_labels[i] if l != config.SUBWORD_TAG_ID]
                all_preds.append(pred_tags)
                all_labels.append(label_tags)

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return avg_loss, f1, precision, recall


def run_training():
    """Hàm chính để chạy toàn bộ quá trình huấn luyện."""
    # --- 1. Thiết lập ---
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tạo thư mục lưu model nếu chưa tồn tại
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # --- 2. Tải Tokenizer và Model ---
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.PRE_TRAINED_MODEL_NAME,
        num_labels=len(config.UNIQUE_TAGS),
        id2label=config.IDS_TO_TAGS,
        label2id=config.TAGS_TO_IDS
    )
    model.to(device)

    # --- 3. Chuẩn bị Dữ liệu ---
    train_dataset = NerDataset(
        file_path=config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        tags_to_ids=config.TAGS_TO_IDS
    )
    dev_dataset = NerDataset(
        file_path=config.DEV_FILE,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        tags_to_ids=config.TAGS_TO_IDS
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.VALID_BATCH_SIZE)

    # --- 4. Optimizer và Scheduler ---
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    num_training_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # --- 5. Vòng lặp Huấn luyện ---
    best_f1 = 0
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_f1, val_precision, val_recall = evaluate(model, dev_dataloader, device, config.IDS_TO_TAGS)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

        # Lưu lại model tốt nhất dựa trên F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1 score: {best_f1:.4f}. Saving model...")
            model.save_pretrained(config.MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
    
    print("\nTraining finished!")
    print(f"Best F1 score on validation set: {best_f1:.4f}")
    print(f"Model saved to {config.MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    run_training()
