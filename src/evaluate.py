# src/evaluate.py
#
# Script để đánh giá hiệu năng cuối cùng của mô hình trên tập test.
# Script này sẽ:
# 1. Tải mô hình và tokenizer đã được huấn luyện tốt nhất.
# 2. Tải và chuẩn bị dữ liệu từ file test.
# 3. Chạy suy luận (inference) và tính toán các chỉ số (metrics).

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from seqeval.metrics import classification_report

# Import các module tự định nghĩa
import config
from dataset import NerDataset

def run_evaluation():
    """Hàm chính để chạy toàn bộ quá trình đánh giá."""
    # --- 1. Thiết lập ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from: {config.MODEL_OUTPUT_DIR}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU. This may be slow.")


    # --- 2. Tải Tokenizer và Model đã lưu ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_OUTPUT_DIR)
        model = AutoModelForTokenClassification.from_pretrained(config.MODEL_OUTPUT_DIR)
        model.to(device)
        model.eval() # Chuyển model sang chế độ đánh giá
    except OSError:
        print(f"Lỗi: Không tìm thấy model tại '{config.MODEL_OUTPUT_DIR}'.")
        print("Vui lòng chạy script 'src/train.py' trước để huấn luyện và lưu model.")
        return

    # --- 3. Chuẩn bị Dữ liệu Test ---
    test_dataset = NerDataset(
        file_path=config.TEST_FILE,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        tags_to_ids=config.TAGS_TO_IDS
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)

    # --- 4. Chạy Đánh giá ---
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Không cần tính gradient khi đánh giá
        for batch in tqdm(test_dataloader, desc="Evaluating on Test Set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            # Chuyển đổi ID sang Tag và xử lý subword để so sánh
            for i in range(len(true_labels)):
                pred_tags = [config.IDS_TO_TAGS[p] for p, l in zip(predictions[i], true_labels[i]) if l != config.SUBWORD_TAG_ID]
                label_tags = [config.IDS_TO_TAGS[l] for l in true_labels[i] if l != config.SUBWORD_TAG_ID]
                
                # Đảm bảo độ dài bằng nhau sau khi loại bỏ subword
                min_len = min(len(pred_tags), len(label_tags))
                all_preds.append(pred_tags[:min_len])
                all_labels.append(label_tags[:min_len])

    # --- 5. In Kết quả ---
    report = classification_report(all_labels, all_preds, digits=4)
    print("\n--- Final Evaluation Report on Test Set ---")
    print(report)


if __name__ == "__main__":
    run_evaluation()
