"""Public resources and constants."""

import logging
import pathlib
from enum import Enum
from typing import Annotated

from app.config import settings
from app.io import (
    InvalidRangeError,
    RangeOutOfBoundsError,
    is_file_readable,
    send_partial_file,
)
from fastapi import APIRouter, Header, HTTPException, Path, Query, status
from fastapi.responses import FileResponse

from ..models.antibiotics import ANTIBIOTICS
from ..models.qc import VARIANT_REJECTION_REASONS

LOG = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_TAGS = [
    "resources",
]
READ_PERMISSION = "resources:read"


@router.get("/resources/antibiotics", tags=DEFAULT_TAGS)
async def get_antibiotics():
    """Get antibiotic names."""
    return ANTIBIOTICS


@router.get("/resources/variant/rejection", tags=DEFAULT_TAGS)
async def get_variant_rejection():
    """Get antibiotic names."""
    return VARIANT_REJECTION_REASONS


class AnnotationType(Enum):
    """Valid annotation file types."""

    FASTA = "fasta"
    FASTA_INDEX = "fasta_index"
    GFF = "gff"


@router.get("/resources/genome/{accession_no}/info", tags=DEFAULT_TAGS)
async def get_genome_resources(
    accession_no: str = Path(..., title="Genome accession number", min_length=3),
    annotation_type: AnnotationType = Query(...),
    range: Annotated[str | None, Header()] = None,
) -> str:
    """Genome sequence and annotated genes for a given reference genome."""
    base_path = pathlib.Path(settings.reference_genomes_dir)
    match annotation_type:
        case AnnotationType.FASTA:
            suffix = ".fasta"
        case AnnotationType.FASTA_INDEX:
            suffix = ".fasta.fai"
        case AnnotationType.GFF:
            suffix = ".gff"

    # generate file path
    file_path = base_path.joinpath(f"{accession_no}{suffix}")

    # check if resource exist
    if not is_file_readable(str(file_path)):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occured reading {accession_no} data",
        )

    # send file if byte range is not set
    if range is None:
        response = FileResponse(file_path, filename=file_path.name)
    else:
        try:
            response = send_partial_file(file_path, range)
        except InvalidRangeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except RangeOutOfBoundsError as error:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail=str(error),
            ) from error
    return response


@router.get("/resources/genome/{accession_no}/annotation", tags=DEFAULT_TAGS)
async def get_genome_annotation(
    accession_no: str = Path(..., title="Genome accession number", min_length=3),
    file: str = Query(..., description="Name of the annotation file with suffix"),
    range: Annotated[str | None, Header()] = None,
) -> str:
    """Get additional genomic annotation for a reference genome."""
    # check if resource exist
    file_path = pathlib.Path(settings.annotations_dir).joinpath(file)
    if not is_file_readable(str(file_path)):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occured reading {accession_no} data",
        )

    # send file if byte range is not set
    if range is None:
        response = FileResponse(file_path, filename=file_path.name)
    else:
        try:
            response = send_partial_file(file_path, range)
        except InvalidRangeError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except RangeOutOfBoundsError as error:
            raise HTTPException(
                status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                detail=str(error),
            ) from error
    return response
