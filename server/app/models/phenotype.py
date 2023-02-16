"""Datamodels used for prediction results."""
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from .base import RWModel


class VariantType(Enum):
    SUBSTITUTION = "substitution"


class DatabaseReference(RWModel):
    ref_database: str
    ref_id: str


class GeneBase(BaseModel):
    """Container for gene information"""

    name: str
    accession: str
    # prediction info
    depth: float | None
    identity: float
    coverage: float
    ref_start_pos: int
    ref_end_pos: int
    ref_gene_length: int
    alignment_length: int


class ResistanceGene(GeneBase, DatabaseReference):
    """Container for resistance gene information"""

    phenotypes: List[str]


class VirulenceGene(GeneBase, DatabaseReference):
    """Container for virulence gene information"""

    virulence_category: str


class VariantBase(DatabaseReference):
    """Container for mutation information"""

    variant_type: VariantType = Field(
        ..., alias="variantType"
    )  # type of mutation insertion/deletion/substitution
    genes: List[str]
    position: int
    ref_codon: str = Field(..., alias="refCodon")
    alt_codon: str = Field(..., alias="altCodon")
    # prediction info
    depth: float


class ResistanceVariant(VariantBase):
    """Container for resistance variant information"""

    phenotypes: List[str]


class PhenotypeResult(BaseModel):
    """Phenotyp result data model.

    A phenotyp result is a generic data structure that stores predicted genes,
    mutations and phenotyp changes.
    """

    phenotypes: Dict[str, List[str]]
    genes: List[ResistanceGene | VirulenceGene]
    mutations: List[ResistanceVariant]


class PhenotypeType(Enum):
    """Valid phenotyp methods."""

    AMR = "antimicrobial_resistance"
    CHEM = "chemical_resistance"
    ENV = "environmental_factor_resistance"
    VIR = "virulence"


class PhenotypeResultIndex(RWModel):
    """Basic key-value index for analysis results."""

    type: PhenotypeType
    result: PhenotypeResult
