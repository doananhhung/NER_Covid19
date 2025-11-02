# src/entity_structures.py
#
# Định nghĩa các cấu trúc dữ liệu cho Phương án 4

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class Entity:
    """
    Đại diện cho một entity được trích xuất từ NER model
    """
    text: str
    tag: str
    start: int
    end: int
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        return f"Entity('{self.text}', {self.tag}, [{self.start}:{self.end}])"
    
    def center(self) -> float:
        """Tính vị trí trung tâm của entity"""
        return (self.start + self.end) / 2
    
    def length(self) -> int:
        """Độ dài của entity"""
        return self.end - self.start


@dataclass
class PatientRecord:
    """
    Đại diện cho thông tin của một bệnh nhân
    """
    # Thông tin nhận dạng
    patient_id: Optional[str] = None
    name: Optional[str] = None
    
    # Thông tin cá nhân
    age: Optional[str] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    
    # Thông tin thời gian (phân loại theo loại sự kiện)
    dates: Dict[str, List[str]] = field(default_factory=lambda: {
        'admission_date': [],
        'test_date': [],
        'positive_date': [],
        'negative_date': [],
        'discharge_date': [],
        'entry_date': [],
        'recovery_date': [],
        'death_date': [],
        'unknown_date': []
    })
    
    # Thông tin địa điểm & tổ chức
    locations: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    transportations: List[str] = field(default_factory=list)
    
    # Thông tin y tế
    symptoms_and_diseases: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 1.0
    assigned_entities: List[Entity] = field(default_factory=list)
    position_range: Tuple[int, int] = (0, 0)
    warnings: List[str] = field(default_factory=list)
    raw_text_snippet: str = ""
    
    # Properties để truy cập dates dễ dàng hơn
    @property
    def admission_dates(self) -> List[str]:
        return self.dates['admission_date']
    
    @admission_dates.setter
    def admission_dates(self, value: List[str]):
        self.dates['admission_date'] = value
    
    @property
    def discharge_dates(self) -> List[str]:
        return self.dates['discharge_date']
    
    @discharge_dates.setter
    def discharge_dates(self, value: List[str]):
        self.dates['discharge_date'] = value
    
    @property
    def test_dates(self) -> List[str]:
        return self.dates['test_date']
    
    @test_dates.setter
    def test_dates(self, value: List[str]):
        self.dates['test_date'] = value
    
    @property
    def positive_dates(self) -> List[str]:
        return self.dates['positive_date']
    
    @positive_dates.setter
    def positive_dates(self, value: List[str]):
        self.dates['positive_date'] = value
    
    @property
    def negative_dates(self) -> List[str]:
        return self.dates['negative_date']
    
    @negative_dates.setter
    def negative_dates(self, value: List[str]):
        self.dates['negative_date'] = value
    
    @property
    def entry_dates(self) -> List[str]:
        return self.dates['entry_date']
    
    @entry_dates.setter
    def entry_dates(self, value: List[str]):
        self.dates['entry_date'] = value
    
    @property
    def recovery_dates(self) -> List[str]:
        return self.dates['recovery_date']
    
    @recovery_dates.setter
    def recovery_dates(self, value: List[str]):
        self.dates['recovery_date'] = value
    
    @property
    def death_dates(self) -> List[str]:
        return self.dates['death_date']
    
    @death_dates.setter
    def death_dates(self, value: List[str]):
        self.dates['death_date'] = value
    
    @property
    def other_dates(self) -> List[str]:
        return self.dates['unknown_date']
    
    @other_dates.setter
    def other_dates(self, value: List[str]):
        self.dates['unknown_date'] = value
    
    def __repr__(self) -> str:
        id_str = self.patient_id or self.name or "Unknown"
        return f"PatientRecord({id_str}, conf={self.confidence:.2f})"
    
    def has_minimum_info(self) -> bool:
        """Kiểm tra xem có đủ thông tin tối thiểu không"""
        return bool(self.patient_id or (self.name and (self.age or self.gender)))
    
    def add_entity(self, entity: Entity) -> None:
        """Thêm entity vào record"""
        if entity not in self.assigned_entities:
            self.assigned_entities.append(entity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sang dictionary để export CSV"""
        return {
            'patient_id': self.patient_id or '',
            'name': self.name or '',
            'age': self.age or '',
            'gender': self.gender or '',
            'job': self.job or '',
            'dates': self._format_dates(),
            'locations': '; '.join(self.locations) if self.locations else '',
            'organizations': '; '.join(self.organizations) if self.organizations else '',
            'symptoms_diseases': '; '.join(self.symptoms_and_diseases) if self.symptoms_and_diseases else '',
            'transportations': '; '.join(self.transportations) if self.transportations else '',
            'confidence': f"{self.confidence:.2f}",
            'warnings': '; '.join(self.warnings) if self.warnings else '',
            'raw_text_snippet': self.raw_text_snippet[:200] if self.raw_text_snippet else ''
        }
    
    def _format_dates(self) -> str:
        """Format dates theo cấu trúc rõ ràng"""
        result = []
        for date_type, dates_list in self.dates.items():
            if dates_list:
                formatted = f"{date_type}: {', '.join(dates_list)}"
                result.append(formatted)
        return '; '.join(result) if result else ''
    
    def merge_with(self, other: 'PatientRecord') -> None:
        """Gộp thông tin từ record khác vào"""
        # Merge các trường đơn giản
        if not self.patient_id and other.patient_id:
            self.patient_id = other.patient_id
        if not self.name and other.name:
            self.name = other.name
        if not self.age and other.age:
            self.age = other.age
        if not self.gender and other.gender:
            self.gender = other.gender
        if not self.job and other.job:
            self.job = other.job
        
        # Merge lists (union, loại bỏ trùng lặp)
        self.locations = list(set(self.locations + other.locations))
        self.organizations = list(set(self.organizations + other.organizations))
        self.transportations = list(set(self.transportations + other.transportations))
        self.symptoms_and_diseases = list(set(self.symptoms_and_diseases + other.symptoms_and_diseases))
        
        # Merge dates
        for date_type in self.dates.keys():
            if other.dates[date_type]:
                self.dates[date_type] = list(set(self.dates[date_type] + other.dates[date_type]))
        
        # Merge metadata
        self.confidence = max(self.confidence, other.confidence)
        # Merge assigned_entities (không dùng set vì Entity không hashable)
        for entity in other.assigned_entities:
            if entity not in self.assigned_entities:
                self.assigned_entities.append(entity)
        self.warnings = list(set(self.warnings + other.warnings))
        
        # Update position range
        if other.position_range != (0, 0):
            if self.position_range == (0, 0):
                self.position_range = other.position_range
            else:
                self.position_range = (
                    min(self.position_range[0], other.position_range[0]),
                    max(self.position_range[1], other.position_range[1])
                )
