
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import sys
import os
import re
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from text_processor import get_text_processor

class NERPredictor:
    """
    Lớp đóng gói mô hình NER để thực hiện dự đoán trên văn bản mới.
    """
    def __init__(self, model_path: str, use_word_segmentation: bool = True):
        """
        Hàm khởi tạo.

        Args:
            model_path (str): Đường dẫn đến thư mục chứa mô hình và tokenizer đã lưu.
            use_word_segmentation (bool): Sử dụng word segmentation hay không (mặc định True).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.ids_to_tags = self.model.config.id2label
            print(f"Model loaded successfully from {model_path} on device {self.device}")
        except OSError:
            print(f"Lỗi: Không tìm thấy model tại '{model_path}'.")
            self.model = None
        
        self.use_word_segmentation = use_word_segmentation
        self.text_processor = get_text_processor() if use_word_segmentation else None
        
        if use_word_segmentation:
            if self.text_processor and self.text_processor.is_available():
                print("VnCoreNLP word segmentation đã sẵn sàng")
            else:
                print("Cảnh báo: VnCoreNLP không khả dụng. Word segmentation sẽ bị vô hiệu hóa.")
                self.use_word_segmentation = False
    
    def segment_text(self, text: str) -> str:
        """
        Tách từ tiếng Việt sử dụng VnCoreNLP.
        
        Args:
            text (str): Văn bản đầu vào.
            
        Returns:
            str: Văn bản đã được tách từ (các từ ghép nối bằng dấu _)
        """
        if not self.use_word_segmentation or not self.text_processor:
            return text
        
        segmented = self.text_processor.segment_text(text)
        return segmented if segmented is not None else text

    def predict(self, sentence: str, max_length: int = 220, show_debug: bool = False):
        """
        Dự đoán các thực thể trong một câu. Tự động xử lý văn bản dài hơn giới hạn của model.

        Args:
            sentence (str): Câu văn bản đầu vào (chưa segment).
            max_length (int): Độ dài tối đa của mỗi chunk (mặc định 220 tokens).
            show_debug (bool): Hiển thị thông tin debug hay không.

        Returns:
            list: Một danh sách các dictionary, mỗi dictionary chứa thông tin về một thực thể (text, tag, start, end).
        """
        if not self.model:
            print("Model chưa được tải. Không thể dự đoán.")
            return []
        
        # KIỂM TRA VĂN BẢN ĐẦU VÀO
        print(f"\n{'='*80}")
        print(f" VĂN BẢN ĐẦU VÀO:")
        print(f"   - Độ dài: {len(sentence)} ký tự")
        print(f"   - Số từ: {len(sentence.split())} từ")
        print(f"   - 100 ký tự đầu: {sentence[:100]}...")
        print(f"   - 100 ký tự cuối: ...{sentence[-100:]}")
        print(f"{'='*80}\n")
        
        # Lưu văn bản gốc (chưa segment) để tìm vị trí entities
        original_text = sentence
        
        # Áp dụng word segmentation nếu được bật
        if self.use_word_segmentation:
            if show_debug:
                print(f" Original text: {sentence[:100]}...")
            sentence_segmented = self.segment_text(sentence)
            if show_debug:
                print(f" Segmented text: {sentence_segmented[:100]}...")
        else:
            sentence_segmented = sentence

        # Kiểm tra độ dài văn bản
        tokens = self.tokenizer.tokenize(sentence_segmented)
        
        print(f"\n Thông tin xử lý:")
        print(f"   - Độ dài văn bản gốc: {len(original_text)} ký tự")
        print(f"   - Độ dài văn bản đã segment: {len(sentence_segmented)} ký tự")
        print(f"   - Số tokens: {len(tokens)}")
        print(f"   - Max length: {max_length}")
        
        if len(tokens) <= max_length:
            # Văn bản ngắn - predict trực tiếp
            print(f"    Xử lý trực tiếp (văn bản ngắn)")
            return self._predict_single(sentence_segmented, show_debug=show_debug, 
                                       original_text=original_text, text_offset=0)
        else:
            # Văn bản dài - chia nhỏ và predict
            print(f"     Chia thành chunks (văn bản dài: {len(tokens)} > {max_length} tokens)")
            return self._predict_long_text(sentence_segmented, max_length, show_debug=show_debug,
                                          original_text=original_text)

    def _predict_single(self, sentence: str, show_debug: bool = False, original_text: str = None, text_offset: int = 0):
        """
        Dự đoán các thực thể cho văn bản ngắn (≤ max_length tokens).

        Args:
            sentence (str): Câu văn bản đầu vào.
            show_debug (bool): Hiển thị thông tin debug hay không.
            original_text (str): Văn bản gốc để tìm vị trí chính xác (cho văn bản dài).
            text_offset (int): Offset của sentence trong original_text.

        Returns:
            list: Danh sách các entity với thông tin text, tag, start, end.
        """
        # 1. Tokenization - KHÔNG TRUNCATE để tránh mất dữ liệu
        encoding = self.tokenizer(
            sentence, 
            return_tensors="pt",
            truncation=False,  # QUAN TRỌNG: Không cắt văn bản
            padding=False
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Kiểm tra độ dài và cảnh báo nếu quá dài
        actual_length = input_ids.shape[1]
        if actual_length > 256:
            print(f"\n  CẢNH BÁO: Sentence có {actual_length} tokens, vượt quá max_length của model (256)!")
            print(f"   Đang CẮT BỎ phần còn lại. Hãy dùng _predict_long_text() thay thế!")
            # Truncate thủ công
            input_ids = input_ids[:, :256]
            attention_mask = attention_mask[:, :256]

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        # Lấy các token và dự đoán tương ứng
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
        predicted_tags = [self.ids_to_tags[p] for p in predictions]
        
        # Debug: In ra tokens và predicted tags để kiểm tra
        if show_debug:
            print("\n=== DEBUG INFO ===")
            print(f"Sentence: {sentence}")
            print(f"Number of tokens: {len(tokens)}")
            print("Tokens and Tags:")
            for i, (token, tag) in enumerate(zip(tokens, predicted_tags)):
                print(f"  {i}: '{token}' -> {tag}")
            print("==================\n")

        # 3. Post-processing: Nhóm các sub-word token thành thực thể hoàn chỉnh
        # Xử lý đúng với PhoBERT BPE tokenizer (@@)
        entities = []
        current_entity_tokens = []
        current_entity_token_indices = []  # Lưu vị trí của các token
        current_entity_tag = ""

        def merge_tokens_to_text(token_list):
            """
            Ghép các BPE tokens thành text gốc sử dụng tokenizer.
            Đây là cách ĐÚNG để xử lý PhoBERT tokens.
            """
            if not token_list:
                return ""
            
            # Sử dụng tokenizer để convert chính xác
            # Tạo một chuỗi giả để tokenizer xử lý
            text = self.tokenizer.convert_tokens_to_string(token_list)
            
            return text.strip()

        # Bỏ qua token [CLS] ở đầu và [SEP] ở cuối
        for i in range(1, len(predicted_tags) - 1):
            token = tokens[i]
            tag = predicted_tags[i]
            
            # Kiểm tra xem token hiện tại có phải là phần tiếp theo của BPE không
            # PhoBERT BPE: token trước kết thúc bằng @@ thì token hiện tại là phần tiếp theo
            is_bpe_continuation = (i > 1 and tokens[i-1].endswith("@@"))

            if tag.startswith("B-"):
                # TRƯỜNG HỢP ĐẶC BIỆT: Nếu token này là phần tiếp theo của BPE 
                # và đang có entity, thì ưu tiên merge vào entity hiện tại
                if is_bpe_continuation and current_entity_tokens:
                    current_entity_tokens.append(token)
                    current_entity_token_indices.append(i)
                    if show_debug:
                        print(f"  [BPE-FIX] Merged '{token}' (B-{tag[2:]}) into current entity '{current_entity_tag}' due to BPE continuation")
                else:
                    # Nếu đang có một thực thể, lưu nó lại trước khi bắt đầu thực thể mới
                    if current_entity_tokens:
                        entity_text = merge_tokens_to_text(current_entity_tokens)
                        if entity_text:
                            entities.append({
                                "text": entity_text, 
                                "tag": current_entity_tag,
                                "token_indices": current_entity_token_indices.copy()
                            })
                    
                    # Bắt đầu một thực thể mới
                    current_entity_tokens = [token]
                    current_entity_token_indices = [i]
                    current_entity_tag = tag[2:]  # Lấy tên tag (ví dụ: LOCATION từ B-LOCATION)
            
            elif tag.startswith("I-"):
                # TRƯỜNG HỢP 1: Khớp với tag hiện tại
                if current_entity_tag == tag[2:]:
                    current_entity_tokens.append(token)
                    current_entity_token_indices.append(i)
                # TRƯỜNG HỢP 2: Token này là phần tiếp theo của BPE
                elif is_bpe_continuation and current_entity_tokens:
                    # Thêm vào entity hiện tại ngay cả khi tag không khớp (sửa lỗi mô hình)
                    current_entity_tokens.append(token)
                    current_entity_token_indices.append(i)
                    if show_debug:
                        print(f"  [BPE-FIX] Merged '{token}' (I-{tag[2:]}) into current entity '{current_entity_tag}' due to BPE continuation")
                else:
                    # Tag không khớp và không phải phần tiếp theo, lưu entity cũ
                    if current_entity_tokens:
                        entity_text = merge_tokens_to_text(current_entity_tokens)
                        if entity_text:
                            entities.append({
                                "text": entity_text, 
                                "tag": current_entity_tag,
                                "token_indices": current_entity_token_indices.copy()
                            })
                    # Bắt đầu entity mới với tag I- này (xử lý trường hợp thiếu B-)
                    current_entity_tokens = [token]
                    current_entity_token_indices = [i]
                    current_entity_tag = tag[2:]
            
            else: # Tag là 'O'
                # TRƯỜNG HỢP ĐẶC BIỆT: Nếu token này là phần tiếp theo của BPE và đang trong entity
                # thì ưu tiên merge vào entity (vì model có thể tag sai)
                if is_bpe_continuation and current_entity_tokens:
                    current_entity_tokens.append(token)
                    current_entity_token_indices.append(i)
                    if show_debug:
                        print(f"  [BPE-FIX] Merged '{token}' (O) into current entity '{current_entity_tag}' due to BPE continuation")
                else:
                    # Nếu đang có một thực thể, lưu nó lại
                    if current_entity_tokens:
                        entity_text = merge_tokens_to_text(current_entity_tokens)
                        if entity_text:
                            entities.append({
                                "text": entity_text, 
                                "tag": current_entity_tag,
                                "token_indices": current_entity_token_indices.copy()
                            })
                    
                    # Reset
                    current_entity_tokens = []
                    current_entity_token_indices = []
                    current_entity_tag = ""
        
        # Lưu lại thực thể cuối cùng nếu nó kéo dài đến hết câu
        if current_entity_tokens:
            entity_text = merge_tokens_to_text(current_entity_tokens)
            if entity_text:
                entities.append({
                    "text": entity_text, 
                    "tag": current_entity_tag,
                    "token_indices": current_entity_token_indices.copy()
                })

        # Thêm thông tin vị trí (start, end) cho mỗi entity bằng cách tìm trong câu gốc
        entities_with_positions = []
        used_positions = []  # Theo dõi vị trí đã sử dụng để tránh trùng lặp
        
        # Xác định text để search
        search_text = original_text if original_text else sentence
        base_offset = text_offset if original_text else 0
        
        for entity in entities:
            entity_text = entity["text"]
            entity_tag = entity["tag"]
            
            # Convert entity_text từ dạng segmented về dạng original
            # Ví dụ: "Trung_tâm" -> "Trung tâm"
            entity_text_original = entity_text.replace('_', ' ')
            
            # Chiến lược tìm kiếm entity trong text gốc:
            # 1. Tìm trong vùng hiện tại (sentence) với offset
            # 2. Mở rộng vùng tìm kiếm nếu không thấy
            
            # Bước 1: Tìm trong vùng gần text_offset (ưu tiên)
            search_start = max(0, base_offset)
            search_end = min(len(search_text), base_offset + len(sentence) + 100)
            search_region = search_text[search_start:search_end]
            
            found = False
            # Tìm tất cả các vị trí xuất hiện của entity_text_original trong search_region
            search_pos = 0
            while True:
                pos = search_region.find(entity_text_original, search_pos)
                if pos == -1:
                    break
                
                # Tính vị trí tuyệt đối
                absolute_pos = search_start + pos
                absolute_end = absolute_pos + len(entity_text_original)
                
                # Kiểm tra xem vị trí này đã được sử dụng chưa
                is_used = any(absolute_pos < used_end and absolute_end > used_start 
                             for used_start, used_end in used_positions)
                
                if not is_used:
                    entities_with_positions.append({
                        "text": entity_text_original,  # Sử dụng text gốc (không có _)
                        "tag": entity_tag,
                        "start": absolute_pos,
                        "end": absolute_end
                    })
                    used_positions.append((absolute_pos, absolute_end))
                    found = True
                    break
                
                # Tiếp tục tìm ở vị trí tiếp theo
                search_pos = pos + 1
            
            if not found:
                # Bước 2: Mở rộng tìm kiếm trong toàn bộ văn bản
                search_pos = 0
                while True:
                    pos = search_text.find(entity_text_original, search_pos)
                    if pos == -1:
                        break
                    
                    absolute_end = pos + len(entity_text_original)
                    is_used = any(pos < used_end and absolute_end > used_start 
                                 for used_start, used_end in used_positions)
                    
                    if not is_used:
                        entities_with_positions.append({
                            "text": entity_text_original,  # Sử dụng text gốc (không có _)
                            "tag": entity_tag,
                            "start": pos,
                            "end": absolute_end
                        })
                        used_positions.append((pos, absolute_end))
                        found = True
                        break
                    
                    search_pos = pos + 1
            
            if not found:
                # Vẫn không tìm thấy - thêm vào nhưng không có vị trí
                if show_debug:
                    print(f"    Could not find position for entity: '{entity_text_original}' ({entity_tag})")
                entities_with_positions.append({
                    "text": entity_text_original,  # Sử dụng text gốc (không có _)
                    "tag": entity_tag,
                    "start": -1,
                    "end": -1
                })

        # Gộp các NAME entities liên tiếp lại với nhau (post-processing)
        entities_with_positions = self._merge_consecutive_names(entities_with_positions, search_text)
        
        return entities_with_positions

    def _merge_consecutive_names(self, entities: List[Dict[str, any]], text: str) -> List[Dict[str, any]]:
        """
        Gộp các NAME entities liên tiếp thành một entity duy nhất.
        Ví dụ: ["Nguyễn", "Văn", "An"] -> ["Nguyễn Văn An"]
        
        Args:
            entities: Danh sách entities đã có vị trí
            text: Văn bản gốc để kiểm tra
            
        Returns:
            Danh sách entities đã được gộp
        """
        if not entities:
            return entities
        
        # Sắp xếp theo vị trí
        sorted_entities = sorted(entities, key=lambda x: (x.get('start', -1), x.get('end', -1)))
        
        merged = []
        i = 0
        
        while i < len(sorted_entities):
            current = sorted_entities[i]
            
            # Nếu không phải NAME hoặc không có vị trí, giữ nguyên
            if current['tag'] != 'NAME' or current.get('start', -1) == -1:
                merged.append(current)
                i += 1
                continue
            
            # Tìm các NAME entities liên tiếp
            consecutive_names = [current]
            j = i + 1
            
            while j < len(sorted_entities):
                next_entity = sorted_entities[j]
                
                # Kiểm tra xem có phải NAME và có liên tiếp không
                if next_entity['tag'] != 'NAME' or next_entity.get('start', -1) == -1:
                    break
                
                # Kiểm tra khoảng cách giữa 2 entities
                gap = next_entity['start'] - consecutive_names[-1]['end']
                
                # Nếu khoảng cách <= 2 ký tự (khoảng trắng, dấu phẩy), coi là liên tiếp
                if gap <= 2:
                    # Kiểm tra xem giữa 2 entities có gì không
                    between_text = text[consecutive_names[-1]['end']:next_entity['start']]
                    # Chỉ chấp nhận khoảng trắng hoặc không có gì
                    if between_text.strip() in ['', ',']:
                        consecutive_names.append(next_entity)
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Nếu có nhiều hơn 1 NAME liên tiếp, gộp lại
            if len(consecutive_names) > 1:
                start_pos = consecutive_names[0]['start']
                end_pos = consecutive_names[-1]['end']
                full_name = text[start_pos:end_pos].strip()
                
                # Loại bỏ dấu phẩy cuối nếu có
                full_name = full_name.rstrip(',')
                
                merged.append({
                    'text': full_name,
                    'tag': 'NAME',
                    'start': start_pos,
                    'end': start_pos + len(full_name)
                })
            else:
                # Chỉ có 1 NAME, giữ nguyên (nhưng loại bỏ dấu phẩy nếu có)
                name_text = current['text'].rstrip(',')
                merged.append({
                    'text': name_text,
                    'tag': 'NAME',
                    'start': current['start'],
                    'end': current['start'] + len(name_text)
                })
            
            i = j
        
        return merged
    
    def _split_sentences(self, text: str) -> List[Dict[str, any]]:
        """
        Chia văn bản thành các câu với thông tin offset.

        Args:
            text (str): Văn bản đầu vào.

        Returns:
            list: Danh sách các dict chứa {'text': câu, 'start': vị trí bắt đầu, 'end': vị trí kết thúc}
        """
        # Regex để chia câu theo dấu chấm, chấm hỏi, chấm than
        # Giữ lại dấu câu
        sentence_pattern = r'([^.!?]*[.!?]+)|([^.!?]+$)'
        matches = re.finditer(sentence_pattern, text)
        
        sentences = []
        for match in matches:
            sentence_text = match.group(0).strip()
            if sentence_text:
                sentences.append({
                    'text': sentence_text,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Nếu không tìm thấy câu nào, trả về toàn bộ text
        if not sentences:
            sentences = [{'text': text, 'start': 0, 'end': len(text)}]
        
        return sentences

    def _create_chunks(self, text: str, max_length: int, overlap: int = 30) -> List[Dict[str, any]]:
        """
        Chia văn bản thành các chunks với overlap.

        Args:
            text (str): Văn bản đầu vào.
            max_length (int): Độ dài tối đa của mỗi chunk (tính theo tokens).
            overlap (int): Số tokens overlap giữa các chunks.

        Returns:
            list: Danh sách các dict chứa {'text': chunk text, 'start': offset bắt đầu}
        """
        # Tokenize toàn bộ văn bản
        tokens = self.tokenizer.tokenize(text)
        
        chunks = []
        start_idx = 0
        char_position = 0  # Theo dõi vị trí ký tự trong text gốc
        
        while start_idx < len(tokens):
            # Lấy chunk từ start_idx đến start_idx + max_length
            end_idx = min(start_idx + max_length, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Convert tokens về text
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            
            # Tìm vị trí chính xác của chunk_text trong text gốc bắt đầu từ char_position
            chunk_start = text.find(chunk_text.strip(), char_position)
            
            # Nếu không tìm thấy chính xác, sử dụng char_position hiện tại
            if chunk_start == -1:
                chunk_start = char_position
            
            chunks.append({
                'text': chunk_text.strip(),
                'start': chunk_start,
                'token_start': start_idx,
                'token_end': end_idx
            })
            
            # Cập nhật char_position cho chunk tiếp theo
            char_position = chunk_start + len(chunk_text.strip())
            
            # Di chuyển start_idx, trừ đi overlap để tạo vùng chồng lấn
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - overlap
        
        return chunks

    def _predict_long_text(self, text: str, max_length: int, show_debug: bool = False, 
                          original_text: str = None) -> List[Dict[str, any]]:
        """
        Dự đoán các thực thể cho văn bản dài bằng cách chia thành chunks.

        Args:
            text (str): Văn bản đầu vào (đã được segment).
            max_length (int): Độ dài tối đa của mỗi chunk.
            show_debug (bool): Hiển thị thông tin debug hay không.
            original_text (str): Văn bản gốc (chưa segment) để tìm vị trí chính xác.

        Returns:
            list: Danh sách các entity đã được gộp và loại bỏ trùng lặp.
        """
        # Nếu không có original_text, dùng text hiện tại
        if original_text is None:
            original_text = text
        
        # 1. Thử chia theo câu trước
        sentences = self._split_sentences(text)
        
        all_entities = []
        
        # 2. Xử lý từng câu hoặc nhóm câu
        current_batch = ""
        current_batch_start = 0
        
        for i, sent_info in enumerate(sentences):
            sentence = sent_info['text']
            sent_tokens = self.tokenizer.tokenize(sentence)
            
            # Nếu câu này quá dài, chia thành chunks
            if len(sent_tokens) > max_length:
                # Xử lý batch hiện tại trước (nếu có)
                if current_batch:
                    print(f"  📦 Processing batch (offset: {current_batch_start})...")
                    batch_entities = self._predict_single(
                        current_batch, 
                        show_debug=show_debug,
                        original_text=original_text,  # Truyền văn bản gốc chưa segment
                        text_offset=current_batch_start
                    )
                    all_entities.extend(batch_entities)
                    current_batch = ""
                
                # Chia câu dài thành chunks
                print(f"    Sentence too long ({len(sent_tokens)} tokens) - splitting into chunks...")
                chunks = self._create_chunks(sentence, max_length, overlap=30)
                
                for j, chunk in enumerate(chunks):
                    print(f"     Processing chunk {j+1}/{len(chunks)}...")
                    # Tính offset chính xác: vị trí câu trong text + vị trí chunk trong câu
                    chunk_offset = sent_info['start'] + chunk['start']
                    
                    # Debug info
                    if show_debug:
                        print(f"      Chunk offset: {chunk_offset}")
                        print(f"      Chunk text: {chunk['text'][:50]}...")
                    
                    chunk_entities = self._predict_single(
                        chunk['text'], 
                        show_debug=show_debug,
                        original_text=original_text,  # Truyền văn bản gốc chưa segment
                        text_offset=chunk_offset
                    )
                    all_entities.extend(chunk_entities)
                
                current_batch_start = sent_info['end']
            else:
                # Thử thêm câu này vào batch
                test_batch = current_batch + " " + sentence if current_batch else sentence
                test_tokens = self.tokenizer.tokenize(test_batch)
                
                if len(test_tokens) <= max_length:
                    # Còn chỗ, thêm vào batch
                    if not current_batch:
                        current_batch_start = sent_info['start']
                    current_batch = test_batch
                else:
                    # Batch đầy, xử lý batch hiện tại
                    if current_batch:
                        print(f"   Processing batch (offset: {current_batch_start})...")
                        batch_entities = self._predict_single(
                            current_batch, 
                            show_debug=show_debug,
                            original_text=original_text,  # Truyền văn bản gốc
                            text_offset=current_batch_start
                        )
                        all_entities.extend(batch_entities)
                    
                    # Bắt đầu batch mới với câu hiện tại
                    current_batch = sentence
                    current_batch_start = sent_info['start']
        
        # Xử lý batch cuối cùng
        if current_batch:
            print(f"   Processing final batch (offset: {current_batch_start})...")
            batch_entities = self._predict_single(
                current_batch, 
                show_debug=show_debug,
                original_text=original_text,  # Truyền văn bản gốc
                text_offset=current_batch_start
            )
            all_entities.extend(batch_entities)
        
        # 3. Loại bỏ entities trùng lặp (từ vùng overlap)
        unique_entities = self._remove_duplicates(all_entities)
        
        print(f" Completed! Found {len(unique_entities)} unique entities.\n")
        
        return unique_entities

    def _remove_duplicates(self, entities: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Loại bỏ các entities trùng lặp dựa trên vị trí và text.

        Args:
            entities (list): Danh sách các entities.

        Returns:
            list: Danh sách entities đã loại bỏ trùng lặp.
        """
        if not entities:
            return []
        
        # Sắp xếp theo vị trí start
        sorted_entities = sorted(entities, key=lambda x: (x.get('start', -1), x.get('end', -1)))
        
        unique = []
        seen = set()
        
        for entity in sorted_entities:
            # Tạo key duy nhất cho entity
            key = (entity['text'].strip(), entity['tag'], entity.get('start', -1))
            
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique

def main():
    """Hàm main để demo cách sử dụng class NERPredictor."""
    print("--- Demo NER Prediction ---")
    
    # Khởi tạo predictor
    predictor = NERPredictor(model_path=config.MODEL_OUTPUT_DIR)

    # Nếu model được tải thành công
    if predictor.model:
        # Câu ví dụ
        sentence1 = "Bệnh nhân Nguyễn Văn An, 50 tuổi, nhập viện tại Bệnh viện Bạch Mai với triệu chứng sốt cao."
        sentence2 = "Bà Trần Thị B quê ở Hà Nội, làm nghề giáo viên."

        print(f"\nCâu 1: '{sentence1}'")
        entities1 = predictor.predict(sentence1)
        print("Các thực thể được nhận dạng:")
        print(entities1)

        print(f"\nCâu 2: '{sentence2}'")
        entities2 = predictor.predict(sentence2)
        print("Các thực thể được nhận dạng:")
        print(entities2)

if __name__ == "__main__":
    main()

