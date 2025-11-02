# src/patient_extraction/manual_extractor.py
"""
Module xử lý trích xuất thông tin bệnh nhân từ NER results (Manual Mode)
Không sử dụng rule-based anchors/zones, chỉ đơn giản group entities theo loại
"""

from typing import List, Dict
from .entity_structures import Entity, PatientRecord


class ManualPatientExtractor:
    """
    Extractor đơn giản cho manual mode
    Giả định: Tất cả entities trong một đoạn văn bản thuộc về 1 bệnh nhân
    """
    
    def __init__(self):
        """Khởi tạo extractor"""
        pass
    
    def extract_from_ner_results(
        self, 
        ner_results: List[Dict],
        raw_text: str = ""
    ) -> PatientRecord:
        """
        Chuyển đổi NER results thành 1 PatientRecord
        
        Args:
            ner_results: List of dicts với format:
                {
                    'text': str,  # hoặc 'word'
                    'tag': str,
                    'start': int,
                    'end': int,
                    'confidence': float (optional)
                }
            raw_text: Văn bản gốc (để lưu snippet)
            
        Returns:
            PatientRecord: Record chứa thông tin bệnh nhân
        """
        # Bước 1: Chuyển đổi NER results → Entity objects
        entities = self._convert_to_entities(ner_results)
        
        # Bước 2: Group entities theo loại tag
        grouped_entities = self._group_entities_by_tag(entities)
        
        # Bước 3: Tạo PatientRecord từ grouped entities
        record = self._build_patient_record(grouped_entities, entities, raw_text)
        
        return record
    
    def _convert_to_entities(self, ner_results: List[Dict]) -> List[Entity]:
        """Chuyển đổi NER dict → Entity objects"""
        entities = []
        for item in ner_results:
            if item['tag'] != 'O':
                # Support cả 'text' và 'word' key
                text = item.get('text') or item.get('word', '')
                entity = Entity(
                    text=text,
                    tag=item['tag'],
                    start=item['start'],
                    end=item['end'],
                    confidence=item.get('confidence', 1.0)
                )
                entities.append(entity)
        return entities
    
    def _group_entities_by_tag(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Group entities theo loại tag
        
        Returns:
            Dict với key là tag type, value là list entities
            Ví dụ: {
                'PATIENT_ID': [...],
                'NAME': [...],
                'AGE': [...],
                ...
            }
        """
        grouped = {}
        for entity in entities:
            tag = entity.tag
            if tag not in grouped:
                grouped[tag] = []
            grouped[tag].append(entity)
        return grouped
    
    def _build_patient_record(
        self, 
        grouped: Dict[str, List[Entity]],
        all_entities: List[Entity],
        raw_text: str
    ) -> PatientRecord:
        """
        Tạo PatientRecord từ grouped entities
        
        Logic đơn giản:
        - Lấy entity đầu tiên của mỗi loại (hoặc merge nếu có nhiều)
        - Gán vào các field tương ứng của PatientRecord
        """
        record = PatientRecord()
        
        # Thông tin nhận dạng
        if 'PATIENT_ID' in grouped:
            record.patient_id = self._merge_texts(grouped['PATIENT_ID'])
        
        if 'NAME' in grouped:
            record.name = self._merge_texts(grouped['NAME'])
        
        # Thông tin cá nhân
        if 'AGE' in grouped:
            record.age = self._merge_texts(grouped['AGE'])
        
        if 'GENDER' in grouped:
            gender_text = self._merge_texts(grouped['GENDER'])
            # Chuẩn hóa giới tính
            record.gender = self._normalize_gender(gender_text)
        
        # Nếu chưa có gender từ NER, thử suy luận từ context
        if not record.gender:
            record.gender = self._infer_gender_from_context(raw_text, record.name or '')
        
        if 'JOB' in grouped:
            record.job = self._merge_texts(grouped['JOB'])
            # Nếu chưa có gender, thử suy luận từ job
            if not record.gender:
                record.gender = self._infer_gender_from_job(record.job)
        
        # Địa điểm & tổ chức
        if 'LOCATION' in grouped:
            record.locations = [e.text for e in grouped['LOCATION']]
        
        if 'ORGANIZATION' in grouped:
            record.organizations = [e.text for e in grouped['ORGANIZATION']]
        
        if 'TRANSPORTATION' in grouped:
            record.transportations = [e.text for e in grouped['TRANSPORTATION']]
        
        # Triệu chứng & bệnh
        if 'SYMPTOM_AND_DISEASE' in grouped:
            record.symptoms_and_diseases = [e.text for e in grouped['SYMPTOM_AND_DISEASE']]
        
        # Xử lý DATE - phân loại theo keyword trong text
        if 'DATE' in grouped:
            self._assign_dates(record, grouped['DATE'], raw_text)
        
        # Metadata
        record.assigned_entities = all_entities
        record.raw_text_snippet = raw_text[:500] if raw_text else ""
        
        # Tính confidence (trung bình)
        if all_entities:
            total_conf = sum(e.confidence for e in all_entities)
            record.confidence = total_conf / len(all_entities)
        
        # Validate & warnings
        self._add_warnings(record)
        
        return record
    
    def _merge_texts(self, entities: List[Entity]) -> str:
        """Merge text của nhiều entities (dùng cho PATIENT_ID, NAME, AGE, etc.)"""
        if not entities:
            return ""
        if len(entities) == 1:
            return entities[0].text
        # Nếu có nhiều entities, merge bằng space
        return " ".join(e.text for e in entities)
    
    def _normalize_gender(self, gender: str) -> str:
        """
        Chuẩn hóa giới tính về dạng chuẩn
        
        Args:
            gender: Giới tính từ NER (có thể là 'nam', 'nữ', 'male', 'female', etc.)
            
        Returns:
            str: 'Nam' hoặc 'Nữ'
        """
        if not gender:
            return ""
        
        gender_lower = gender.lower().strip()
        
        # Mapping các cách viết khác nhau
        male_variants = ['nam', 'male', 'trai', 'boy', 'man', 'nam giới']
        female_variants = ['nữ', 'nu', 'female', 'gái', 'girl', 'woman', 'nữ giới']
        
        if gender_lower in male_variants:
            return "Nam"
        elif gender_lower in female_variants:
            return "Nữ"
        
        # Nếu không match, giữ nguyên nhưng capitalize
        return gender.capitalize()
    
    def _assign_dates(
        self, 
        record: PatientRecord, 
        date_entities: List[Entity],
        raw_text: str
    ):
        """
        Phân loại DATE entities vào các loại date khác nhau
        
        Logic: Tìm keyword xung quanh date entity để xác định loại
        Keywords:
        - nhập viện, vào viện → admission_date
        - xuất viện, ra viện → discharge_date
        - xét nghiệm, test → test_date
        - dương tính, nhiễm → positive_date
        - âm tính, khỏi → negative_date
        - nhập cảnh → entry_date
        - tử vong, chết → death_date
        """
        keywords_map = {
            'admission_date': ['nhập viện', 'vào viện', 'nhập vào', 'đưa vào'],
            'discharge_date': ['xuất viện', 'ra viện', 'ra khỏi', 'về nhà'],
            'test_date': ['xét nghiệm', 'test', 'lấy mẫu', 'khám'],
            'positive_date': ['dương tính', 'nhiễm', 'mắc', 'phát hiện'],
            'negative_date': ['âm tính', 'khỏi bệnh', 'hồi phục'],
            'entry_date': ['nhập cảnh', 'vào cảnh', 'bay từ', 'từ nước'],
            'death_date': ['tử vong', 'qua đời', 'chết'],
            'recovery_date': ['khỏi', 'hồi phục', 'bình phục']
        }
        
        raw_text_lower = raw_text.lower()
        
        for date_entity in date_entities:
            # Lấy context xung quanh date (50 ký tự trước và sau)
            start = max(0, date_entity.start - 50)
            end = min(len(raw_text), date_entity.end + 50)
            context = raw_text_lower[start:end]
            
            # Tìm keyword match
            matched_type = None
            for date_type, keywords in keywords_map.items():
                if any(kw in context for kw in keywords):
                    matched_type = date_type
                    break
            
            # Gán vào record
            if matched_type:
                record.dates[matched_type].append(date_entity.text)
            else:
                record.dates['unknown_date'].append(date_entity.text)
    
    def _infer_gender_from_context(self, text: str, name: str = "") -> str:
        """
        Suy luận giới tính từ context (từ xưng hô, danh xưng)
        
        Args:
            text: Văn bản gốc
            name: Tên bệnh nhân (nếu có)
            
        Returns:
            str: 'Nam', 'Nữ', hoặc '' nếu không xác định được
        """
        text_lower = text.lower()
        
        # Định nghĩa keywords cho nam/nữ
        male_keywords = [
            # Xưng hô
            r'\banh\b', r'\bông\b', r'\bchú\b', r'\bbác\b.*\banh\b',
            # Giới tính trực tiếp
            r'\bnam\b', r'\btrai\b', r'\bcon trai\b', r'\bcháu trai\b',
            # Nghề nghiệp nam tính
            r'\bthầy giáo\b', r'\bthầy\b', r'\bgiáo viên nam\b',
            # Danh xưng
            r'\bông\s+[A-ZĐ]', r'\banh\s+[A-ZĐ]',
        ]
        
        female_keywords = [
            # Xưng hô
            r'\bchị\b', r'\bcô\b', r'\bbà\b', r'\bbác\b.*\bchị\b',
            # Giới tính trực tiếp
            r'\bnữ\b', r'\bgái\b', r'\bcon gái\b', r'\bcháu gái\b',
            # Nghề nghiệp nữ tính
            r'\bcô giáo\b', r'\bgiáo viên nữ\b',
            # Danh xưng
            r'\bchị\s+[A-ZĐ]', r'\bcô\s+[A-ZĐ]', r'\bbà\s+[A-ZĐ]',
        ]
        
        # Đếm số lượng match cho mỗi giới tính
        import re
        male_count = sum(1 for pattern in male_keywords if re.search(pattern, text_lower))
        female_count = sum(1 for pattern in female_keywords if re.search(pattern, text_lower))
        
        # Quyết định dựa trên số lượng match
        if male_count > female_count:
            return "Nam"
        elif female_count > male_count:
            return "Nữ"
        
        # Nếu không xác định được, thử suy luận từ tên
        if name:
            return self._infer_gender_from_name(name)
        
        return ""
    
    def _infer_gender_from_name(self, name: str) -> str:
        """
        Suy luận giới tính từ tên (dựa vào tên đệm/tên thường gặp)
        
        Args:
            name: Tên bệnh nhân
            
        Returns:
            str: 'Nam', 'Nữ', hoặc '' nếu không xác định được
        """
        name_lower = name.lower()
        
        # Tên đệm thường gặp
        male_middle_names = ['văn', 'đức', 'hữu', 'quang', 'minh', 'hoàng', 'tuấn', 'công']
        female_middle_names = ['thị', 'như', 'kim', 'thu', 'hương', 'thanh', 'mai', 'phương']
        
        # Tên thường gặp
        male_first_names = ['anh', 'dũng', 'hùng', 'nam', 'long', 'phong', 'sơn', 'tuấn', 'cường']
        female_first_names = ['hoa', 'lan', 'mai', 'hương', 'linh', 'nga', 'trang', 'thảo', 'anh']
        
        words = name_lower.split()
        
        # Check tên đệm (thường là từ giữa)
        if len(words) >= 2:
            middle = words[-2]
            if middle in male_middle_names:
                return "Nam"
            elif middle in female_middle_names:
                return "Nữ"
        
        # Check tên (từ cuối)
        if len(words) >= 1:
            first = words[-1]
            if first in male_first_names:
                return "Nam"
            elif first in female_first_names:
                # Chú ý: 'anh' có thể là cả nam và nữ, ưu tiên nữ nếu không có thông tin khác
                if first != 'anh':
                    return "Nữ"
        
        return ""
    
    def _infer_gender_from_job(self, job: str) -> str:
        """
        Suy luận giới tính từ nghề nghiệp
        
        Args:
            job: Nghề nghiệp
            
        Returns:
            str: 'Nam', 'Nữ', hoặc '' nếu không xác định được
        """
        if not job:
            return ""
        
        job_lower = job.lower()
        
        # Nghề nghiệp có xu hướng nam
        male_jobs = [
            'thầy giáo', 'thầy', 'kỹ sư', 'lái xe', 'tài xế', 
            'công nhân xây dựng', 'thợ xây', 'thợ điện', 'thợ máy',
            'bác sĩ nam', 'ông', 'anh'
        ]
        
        # Nghề nghiệp có xu hướng nữ
        female_jobs = [
            'cô giáo', 'cô', 'y tá', 'điều dưỡng', 'nữ hộ sinh', 
            'bà', 'chị', 'giúp việc', 'phụ bếp',
            'bác sĩ nữ'
        ]
        
        for male_job in male_jobs:
            if male_job in job_lower:
                return "Nam"
        
        for female_job in female_jobs:
            if female_job in job_lower:
                return "Nữ"
        
        return ""
    
    def _add_warnings(self, record: PatientRecord):
        """Thêm warnings nếu thiếu thông tin quan trọng"""
        if not record.patient_id and not record.name:
            record.warnings.append("Thiếu thông tin nhận dạng (ID hoặc Tên)")
        
        if not record.age and not record.gender:
            record.warnings.append("Thiếu thông tin cá nhân cơ bản (Tuổi/Giới tính)")
        
        # Check nếu không có entity nào
        if not record.assigned_entities:
            record.warnings.append("Không có entity nào được trích xuất")
            record.confidence = 0.0


def extract_single_patient(ner_results: List[Dict], raw_text: str = "") -> PatientRecord:
    """
    Helper function để extract 1 bệnh nhân từ NER results
    
    Args:
        ner_results: List of NER prediction dicts
        raw_text: Original text
        
    Returns:
        PatientRecord
    """
    extractor = ManualPatientExtractor()
    return extractor.extract_from_ner_results(ner_results, raw_text)
