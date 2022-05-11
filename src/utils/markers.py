import numpy as np
from src import pairwise_differences as pwd


def markers_from_solution(
		adata,
		*,
		key,
		solution,
		M=None,
		index_to_pair=None,
		ordered=True,
		threshold=1,
	):
	"""Obtain markers for each phenotype based on genes
	selected in solution.

	Parameters
	__________
	adata: AnnData object
	key: str
		Key in adata.obs
	solution: array-like
		Indices of features to select in adata
	M: None or ndarray
		Pairwise distances matrix
	index_to_pair: dict
		Dictionary converting row indices in M to a pair of phenotypes
	ordered: bool
		Whether to construct an ordered pairwise matrix or not
	threshold: float
		Only features with a coverage of greater than `threshold` will
		be considered as markers for a given phenotype.
	"""
	solution = np.sort(np.asarray(solution))
	assert solution.max() < adata.shape[1]

	# only need to look at genes in the solution
	adata = adata[:, solution]
	unq_phenotypes = np.unique(adata.obs[key])
	M, index_to_pair = pwd(
		adata.X, adata.obs[key], classes=unq_phenotypes, ordered=ordered)

	markers = {i: [] for i in range(unq_phenotypes.size)}

	for ip in index_to_pair:
		i, _ = index_to_pair[ip]
		markers[i].append(np.argwhere(M[ip] >= threshold).flatten())
	for i in range(unq_phenotypes.size):
		markers[i] = np.unique(np.concatenate(markers[i]))

	names = {
		unq_phenotypes[i]: adata.var_names[markers[i]].to_numpy()
		for i in markers
	}
	return names
