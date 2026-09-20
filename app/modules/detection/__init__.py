"""
Detection Module — Blood Pathology Analysis
Supports Malaria, Sickle Cell Disease, ALL, and Iron Deficiency.
"""

from .malaria import MalariaDetector
from .sickle_cell import SickleCellDetector
from .all_leukemia import ALLDetector
from .iron_deficiency import IronDeficiencyDetector

__all__ = ["MalariaDetector", "SickleCellDetector", "ALLDetector", "IronDeficiencyDetector"]