"""Functions that generate data rendered by views."""
import logging
from itertools import chain, groupby
from typing import Any, Dict, Tuple

from app.models import NT_TO_AA, ElementType, PredictionSoftware

LOG = logging.getLogger(__name__)
SampleObj = Dict[str, Any]


def _has_phenotype(feature, phenotypes) -> bool:
    """Check if gene or mutation has phenotype."""
    phenotypes = [phe.lower() for phe in phenotypes]
    return any(pheno.lower() in phenotypes for pheno in feature["phenotypes"])


def filter_validated_genes(validated_genes, sample: SampleObj):
    """Remove genes that have not been validated.

    :param validated_genes: Genes that have been validated
    :type validated_genes: _type_
    :param sample: Sample prediction result
    :type sample: SampleObj
    :return: Validated genes
    """
    for category, valid_genes in validated_genes.items():
        LOG.debug("Removing non-validated genes from input")
        pred_res = next(
            iter([r for r in sample["phenotypeResult"] if r["type"] == category])
        )
        # filter genes based on the list of validated genes/ phenotypes for group
        if category.endswith("resistance"):
            filtered_genes = [
                res
                for res in pred_res["result"]["genes"]
                if _has_phenotype(res, valid_genes)
            ]
            filtered_mutations = [
                res
                for res in pred_res["result"]["mutations"]
                if _has_phenotype(res, valid_genes)
            ]
            resistance = {
                phe
                for feat in chain(filtered_genes, filtered_mutations)
                for phe in feat["phenotypes"]
            }
            # update phenotypes
            pred_res["result"]["phenotypes"] = {
                "resistant": list(resistance),
                "susceptible": list(set(validated_genes) - resistance),
            }
            pred_res["result"]["genes"] = filtered_genes
            pred_res["result"]["mutations"] = filtered_mutations
        else:
            genes = [
                gene
                for gene in pred_res["result"].get("genes", [])
                if gene["name"] in valid_genes
            ]
            mutations = [
                mut
                for mut in pred_res["result"].get("mutations", [])
                if mut["genes"] in valid_genes
            ]
            # update object from database
            pred_res["result"]["genes"] = genes
            pred_res["result"]["mutations"] = mutations
    return pred_res


def create_amr_summary(sample: SampleObj) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Summarize antimicrobial resistance prediction.

    :param sample: Sample information
    :type sample: SampleObj
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: Summary table and resistance information.
    :rtype: Tuple[Dict[str, Any], Dict[str, Any]]
    """
    amr_summary = {}
    resistance_info = {"genes": {}, "mutations": {}}
    LOG.debug("Make AMR prediction summary table")
    for pred_res in sample["element_type_result"]:
        # only get AMR resistance
        if pred_res["type"] == ElementType.AMR.value:
            for gene in pred_res["result"]["genes"]:
                gene_name = gene["gene_symbol"]
                if gene_name is None:
                    raise ValueError
                # get/create summary dictionary object
                gene_entry = amr_summary.get(
                    gene_name,
                    {
                        # create default object
                        "gene_symbol": gene_name,
                        "software": [],
                        "res_class": "Unknown",
                    },
                )
                # annotate softwares
                gene_entry["software"].append(pred_res["software"])

                # annotate resistance class
                if pred_res["software"] == PredictionSoftware.AMRFINDER.value:
                    gene_entry["res_class"] = gene["res_class"]

                # store object
                amr_summary[gene_name] = gene_entry

                # reformat resistance gene table
                gene_entry = resistance_info["genes"].get(gene_name, [])
                gene["software"] = pred_res["software"]
                gene_entry.append(gene)
                resistance_info["genes"][gene_name] = gene_entry

            # iterate over mutations and populate resistance summaries
            for mutation in pred_res["result"]["mutations"]:
                gene_name, *_ = mutation["ref_id"].split(";;")
                # gene entries
                gene_entry = amr_summary.get(
                    gene_name,
                    {
                        # create default object
                        "gene_symbol": gene_name,
                        "software": [],
                        "res_class": "Unknown",
                    },
                )
                if mutation["variant_type"] == "substitution":
                    ref_aa = NT_TO_AA[mutation["ref_codon"].upper()]
                    alt_aa = NT_TO_AA[mutation["alt_codon"].upper()]
                    gene_entry["change"] = f"{ref_aa}{mutation['position']}{alt_aa}"
                else:
                    raise ValueError
                # store object
                amr_summary[gene_name] = gene_entry

    # group summary by res_class
    amr_summary = {
        res_type: list(rows)
        for res_type, rows in groupby(
            amr_summary.values(), key=lambda x: x["res_class"]
        )
    }
    return amr_summary, resistance_info
