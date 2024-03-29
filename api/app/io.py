"""File IO operations."""
import itertools
import logging
import mimetypes
import os
import pathlib
import re
from collections import defaultdict
from typing import List

import pandas as pd
from fastapi.responses import Response
from prp.models.phenotype import GeneBase, PredictionSoftware, VariantBase
from prp.models.typing import TypingMethod
from pydantic import BaseModel

from .models.sample import SampleInDatabase

LOG = logging.getLogger(__name__)
BYTE_RANGE_RE = re.compile(r"bytes=(\d+)-(\d+)?$")

TARGETED_ANTIBIOTICS = {
    "rifampicin": {"abbrev": "rif", "split_res_level": False},
    "isoniazid": {"abbrev": "inh", "split_res_level": True},
    "pyrazinamide": {"abbrev": "pyr", "split_res_level": False},
    "ethambutol": {"abbrev": "etb", "split_res_level": False},
    "amikacin": {"abbrev": "ami", "split_res_level": False},
    "levofloxacin": {"abbrev": "lev", "split_res_level": False},
}


class InvalidRangeError(Exception):
    pass


class RangeOutOfBoundsError(Exception):
    pass


def is_file_readable(file_path: str) -> bool:
    """Check if file exist and is readable.

    :param file_path: File path object
    :type file_path: str
    :return: True if readable and exist
    :rtype: bool
    """
    path = pathlib.Path(file_path)
    if not path.is_file():
        LOG.warning("trying to access missing reference genome data: %s", file_path)
        return False

    if not os.access(path, os.R_OK):
        LOG.warning("file: %s cant read by the system user", file_path)
        return False

    return True


def parse_byte_range(byte_range):
    """Returns the two numbers in 'bytes=123-456' or throws ValueError.
    The last number or both numbers may be None.
    """
    if byte_range.strip() == "":
        return None, None

    m = BYTE_RANGE_RE.match(byte_range)
    if not m:
        raise InvalidRangeError(f"Invalid byte range {byte_range}")

    first, last = [x and int(x) for x in m.groups()]
    if last and last < first:
        raise InvalidRangeError(f"Invalid byte range {byte_range}")
    return first, last


def send_partial_file(path, range_header):
    byte_range = parse_byte_range(range_header)
    first, last = byte_range

    data = None
    with open(path, "rb") as file_handle:
        fs = os.fstat(file_handle.fileno())
        file_len = fs[6]
        if first >= file_len:
            raise RangeOutOfBoundsError("Requested Range Not Satisfiable")

        if last is None or last >= file_len:
            last = file_len - 1
        response_length = last - first + 1

        file_handle.seek(first)
        data = file_handle.read(response_length)

    response_headers = {
        "Content-type": "application/octet-stream",
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {first}-{last}/{file_len}",
        "Content-Length": str(response_length),
    }
    return Response(
        content=data,
        status_code=206,
        media_type=mimetypes.guess_type(path)[0],
        headers=response_headers,
    )


def _sort_motifs_on_phenotype(prediction: List[GeneBase | VariantBase]):
    """Sort resistance genes and variants by the antibiotics they yeid resistance to."""
    result = defaultdict(lambda: defaultdict(list))
    for motif in prediction:
        if len(motif.phenotypes) == 0:
            result["none"]["none"].append(motif)
        else:
            # add variant to all phenotypes
            for phenotype in motif.phenotypes:
                # make a copy with only one phenotypes
                upd_motif = motif.model_copy(update={"phenotypes": [phenotype]})
                result[phenotype.name][phenotype.resistance_level].append(upd_motif)
    return result


def _fmt_variant(variant):
    """"""
    WHO_CLASSES = {
        "Assoc w R": 1,
        "Assoc w R - Interim": 2,
        "Uncertain significance": 3,
        "Not assoc w R - Interim": 4,
        "Not assoc w R": 5,
    }
    var_type = variant.variant_type
    if var_type == 'SV':
        variant_desc = f"{var_type}_{variant.variant_subtype}_{variant.start}-{variant.end}"
    else:
        variant_desc = f"{variant.reference_sequence}_{variant.start}_{variant.variant_subtype}"
    # annotate variant frequency for minority variants
    if variant.frequency is not None and variant.frequency < 1:
        variant_desc = f"{variant_desc}({variant.frequency * 100:.1f}%)"
    # annotate WHO classification
    who_group = WHO_CLASSES.get(variant.phenotypes[0].note)
    if who_group is not None:
        variant_desc = f"{variant_desc} WHO-{who_group}"
    return variant_desc


def _fmt_mtuberculosis(sample: SampleInDatabase):
    """Format M. tuberculosis result.

    - one row per antibiotic with all variants

    :param sample: Prediction results
    :type sample: SampleInDatabase
    """
    result = []
    for pred_res in sample.element_type_result:
        # only include TBprofiler result
        if not pred_res.software == PredictionSoftware.TBPROFILER:
            continue

        # combine tbprofiler variants, called SNV and called SV
        filtered_variants = [
            var
            for var in itertools.chain(
                pred_res.result.variants, sample.sv_variants, sample.snv_variants
            )
            if var.verified == "passed"
        ]
        # reformat predicted resistance
        sorted_variants = _sort_motifs_on_phenotype(filtered_variants)

        # create tabular result
        positive = "Mutation påvisad"
        negative = "Mutation ej påvisad"
        for antibiotic in TARGETED_ANTIBIOTICS:
            # concat variants
            if TARGETED_ANTIBIOTICS[antibiotic]["split_res_level"]:
                for lvl in ["high", "low"]:
                    abbrev = TARGETED_ANTIBIOTICS[antibiotic]["abbrev"]
                    if (
                        antibiotic in sorted_variants
                        and len(sorted_variants[antibiotic][lvl]) > 0
                    ):
                        call = positive
                        variants = ";".join(
                            _fmt_variant(var)
                            for var in sorted_variants[antibiotic][lvl]
                        )
                    else:
                        # add non-called resistance
                        call = negative
                        variants = "-"
                    result.append(
                        {
                            "sample_id": sample.sample_id,
                            "parameter": f"{abbrev.upper()} NGS{lvl[0].upper()}",
                            "result": call,
                            "variants": variants,
                        }
                    )
            else:
                if antibiotic in sorted_variants:
                    call = positive
                    combined_variants = itertools.chain(
                        *sorted_variants[antibiotic].values()
                    )
                    variants = ";".join(_fmt_variant(var) for var in combined_variants)
                else:
                    # add non-called resistance
                    call = negative
                    variants = "-"
                # add result
                result.append(
                    {
                        "sample_id": sample.sample_id,
                        "parameter": f"{TARGETED_ANTIBIOTICS[antibiotic]['abbrev'].upper()} NGS",
                        "result": call,
                        "variants": variants,
                    }
                )
    # annotate species prediction res
    result.append(
        {
            "sample_id": sample.sample_id,
            "parameter": "MTBC ART",
            "result": sample.species_prediction[0].scientific_name,
            "variants": "-",
        }
    )
    # annotate lineage
    for type_res in sample.typing_result:
        if (
            type_res.type == TypingMethod.LINEAGE
            and type_res.software == PredictionSoftware.TBPROFILER
        ):
            # get lineage with longest lineage string
            lin = max(type_res.result.lineages, key=lambda x: len(x.lin))
            result.append(
                {
                    "sample_id": sample.sample_id,
                    "parameter": "MTBC LINEAGE",
                    "result": lin.lin,
                    "variants": "-",
                }
            )
    return pd.DataFrame(result)


def sample_to_kmlims(sample: SampleInDatabase) -> pd.DataFrame:
    """Convert sample information to KMLIMS format."""
    match sample.run_metadata.run.analysis_profile:
        case "mycobacterium_tuberculosis":
            pred_res = _fmt_mtuberculosis(sample)
        case default:
            raise NotImplementedError(
                f"No export function for {sample.run_metadata.run.analysis_profile}"
            )
    return pred_res
