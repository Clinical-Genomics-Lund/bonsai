"""Function for clustering."""
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from ..crud.sample import TypingProfileOutput
from enum import Enum
import pandas as pd
from skbio.tree import nj
from skbio import DistanceMatrix
import sourmash
from app import config
from pathlib import Path
from typing import List


class DistanceMethod(Enum):
    """Index of methods for calculating the distance matrix during hierarchical clustering of samples."""

    JACCARD = "jaccard"
    HAMMING = "hamming"


class ClusterMethod(Enum):
    """Index of methods for hierarchical clustering of samples."""

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    NJ = "neighbor_joining"


def to_newick(node, newick, parentdist, leaf_names):
    """Convert hierarcical tree representation to newick format."""

    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = to_newick(node.get_left(), newick, node.dist, leaf_names)
        newick = to_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
        newick = f"({newick}"
        return newick


def cluster_on_allele_profile(
    profiles: TypingProfileOutput,
    method: ClusterMethod,
    distance_metric: DistanceMethod,
) -> str:
    """Cluster samples on allele profiles."""
    obs = pd.DataFrame(
        [sample.allele_profile() for sample in profiles], 
        index=[sample.sample_id for sample in profiles]
    )
    leaf_names = list(obs.index)
    # calcualte distance matrix
    dist_mtrx = pdist(obs, metric=distance_metric.value)

    if method == ClusterMethod.NJ:
        dm = DistanceMatrix(dist_mtrx, leaf_names)
        newick_tree = nj(dm, result_constructor=str)
    else:
        # to use the clustering methods in scipy
        Z = hierarchy.linkage(dist_mtrx, method=method.value)
        # convert distance matrix to cluster
        tree = hierarchy.to_tree(Z, False)
        newick_tree = to_newick(tree, "", tree.dist, leaf_names)
    return newick_tree


def cluster_on_minhash_signature(sample_ids: List[str], method: ClusterMethod):
    """Cluster multiple samples on their minhash signatures."""
    # load sourmash index
    signature_dir = Path(config.GENOME_SIGNATURE_DIR)

    # load sequence signatures to memory
    siglist = []
    kmer_sizes = config.SIGNATURE_KMER_SIZE
    for sample_id in sample_ids:
        signature_path = signature_dir.joinpath(f"{sample_id}.sig")
        if not signature_path.is_file():
            raise FileNotFoundError(f"Signature file not found, {signature_path}")

        # read signature
        loaded = sourmash.load_file_as_signatures(str(signature_path), 
                                                  ksize=kmer_sizes)
        # check if signatures were fund
        loaded = list(loaded)
        if not loaded:
            raise ValueError(f"No signatures, sample id: {sample_id}, ksize: {kmer_sizes}, {loaded}")
        siglist.extend(loaded)  # append to all signatures

    # create distance matrix
    similarity = sourmash.compare.compare_all_pairs(siglist, 
                                                    ignore_abundance=True, 
                                                    n_jobs=1, 
                                                    return_ani=False)
    # cluster on similarity matrix
    Z = hierarchy.linkage(similarity, method=method.value)
    tree = hierarchy.to_tree(Z, False)
    # creae newick tree
    labeltext = [str(item).replace(".fasta", "") for item in siglist]
    newick_tree = to_newick(tree, "", tree.dist, labeltext)
    return newick_tree