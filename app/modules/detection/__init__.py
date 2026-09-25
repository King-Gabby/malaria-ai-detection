"""
Detection Module — Blood Pathology Analysis
Supports Malaria, Sickle Cell Disease, ALL, and Iron Deficiency.
"""

from .malaria import MalariaDetector, load_malaria_model
from .sickle_cell import SickleCellDetector, load_sickle_cell_model
from .all_leukemia import ALLDetector, load_all_model
from .iron_deficiency import IronDeficiencyDetector, load_iron_deficiency_model

__all__ = [
    "MalariaDetector",
    "SickleCellDetector",
    "ALLDetector",
    "IronDeficiencyDetector",
    "load_malaria_model",
    "load_sickle_cell_model",
    "load_all_model",
    "load_iron_deficiency_model",
]