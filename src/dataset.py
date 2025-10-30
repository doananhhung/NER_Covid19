# src/dataset.py
#
# File này định nghĩa lớp NerDataset, chịu trách nhiệm tải, tiền xử lý
# và chuẩn bị dữ liệu cho việc huấn luyện và đánh giá mô hình NER.

import torch
import pandas as pd
import os

class NerDataset(torch.utils.data.Dataset):
    """
    Lớp Dataset cho bài toán NER.
    Kế thừa từ torch.utils.data.Dataset.
    """
    def __init__(self, file_path, tokenizer, max_len, tags_to_ids):
        """
        Hàm khởi tạo.

        Args:
            file_path (str): Đường dẫn đến file dữ liệu (train/dev/test).
            tokenizer: Tokenizer của Hugging Face (ví dụ: PhoBERT tokenizer).
            max_len (int): Độ dài tối đa của chuỗi sau khi token hóa.
            tags_to_ids (dict): Bảng map từ tên nhãn sang ID.
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tags_to_ids = tags_to_ids
        self.subword_tag_id = -100 # ID đặc biệt để Pytorch bỏ qua khi tính loss
        
        # Đọc và xử lý dữ liệu ngay khi khởi tạo
        self.sentences, self.tags = self._read_data()

    def _read_data(self):
        """
        Đọc dữ liệu từ file JSON Lines và tách thành 2 list: words và tags.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File không được tìm thấy tại: {self.file_path}")
            
        df = pd.read_json(self.file_path, lines=True, encoding='utf-8')
        sentences = df['words'].tolist()
        tags = df['tags'].tolist()
        return sentences, tags

    def __len__(self):
        """
        Trả về tổng số câu trong dataset.
        """
        return len(self.sentences)

    def __getitem__(self, index):
        """
        Lấy một mẫu dữ liệu tại vị trí `index`.
        Đây là nơi logic tiền xử lý chính diễn ra.
        """
        words = self.sentences[index]
        tags = self.tags[index]

        # --- Logic Tokenization và Căn chỉnh Nhãn (Alignment) ---
        input_ids = []
        target_tags = []

        for i, word in enumerate(words):
            # Token hóa từng từ
            word_tokens = self.tokenizer.tokenize(word)

            # Nếu từ bị tách thành các sub-word
            if len(word_tokens) > 0:
                input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_tokens))
                
                # Gán nhãn cho sub-word đầu tiên
                tag_id = self.tags_to_ids.get(tags[i], self.tags_to_ids['O'])
                target_tags.append(tag_id)
                
                # Gán nhãn đặc biệt (-100) cho các sub-word còn lại
                target_tags.extend([self.subword_tag_id] * (len(word_tokens) - 1))

        # --- Padding và Truncating ---
        # Cắt bớt nếu dài hơn max_len (trừ đi 2 cho [CLS] và [SEP])
        if len(input_ids) > self.max_len - 2:
            input_ids = input_ids[:self.max_len - 2]
            target_tags = target_tags[:self.max_len - 2]

        # Thêm các token đặc biệt [CLS] và [SEP]
        final_input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        final_target_tags = [self.subword_tag_id] + target_tags + [self.subword_tag_id]

        # Tạo attention mask
        attention_mask = [1] * len(final_input_ids)

        # Padding đến max_len
        padding_length = self.max_len - len(final_input_ids)
        final_input_ids = final_input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        final_target_tags = final_target_tags + ([self.subword_tag_id] * padding_length)
        
        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(final_target_tags, dtype=torch.long)
        }
