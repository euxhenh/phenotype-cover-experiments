import anndata
import numpy as np
import scanpy as sc


def _combine(arr1, arr2, splitter=':'):
    combined = [str(i) + splitter + str(j) for i, j in zip(arr1, arr2)]
    return np.array(combined)


def dataset(
    name,
    *,
    normalize=False,
    log=False,
    scale=False,
    high_var=False,
    filter_genes=False,
):
    """Loads dataset as an AnnData object and performs any preprocessing.
    """
    if name == "IPF":
        adata = anndata.read_h5ad("data/IPF.h5ad")
        key = 'label'
    elif name == "HCA":
        adata = anndata.read_h5ad('data/HCA.h5ad')
        print(f"{len(np.unique(adata.obs['celltype']))} cell types.")
        print(f"{len(np.unique(adata.obs['tissue']))} tissues.")
        adata.obs['tissue:celltype'] = _combine(adata.obs['tissue'], adata.obs['celltype'])
        print(f"{len(np.unique(adata.obs['tissue:celltype']))} tissue-cell type combinations.")
        key = 'tissue:celltype'
    elif name == "MC":
        adata = anndata.read_h5ad('data/MC.h5ad')
        print(f"{len(np.unique(adata.obs['celltype']))} unique cell types.")
        key = 'celltype'
    else:
        raise ValueError("Dataset not found.")

    print(f"{adata.shape=}")
    if filter_genes:
        sc.pp.filter_genes(adata, min_cells=10)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log:
        sc.pp.log1p(adata)
    # load hv genes from file in case log was False.
    # do this since highly_variable_genes requires logged input.
    if name == "HCA" and high_var:
        hehv = np.genfromtxt('data/he-hv.txt', dtype='str')
        adata = adata[:, hehv]
        high_var = False
    if high_var:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.2)
        adata = adata[:, adata.var.highly_variable]
    if scale:
        sc.pp.scale(adata, max_value=10)

    return adata, key


def remove_low_count_ct(adata, key, thresh=50):
    if key not in adata.obs:
        raise KeyError("Key not found in adata.")
    ct, counts = np.unique(adata.obs[key], return_counts=True)
    ct_rem = ct[counts >= thresh]
    adata = adata[np.isin(adata.obs[key], ct_rem)]

    print(f"Removed low count classes.")
    print(f"{adata.shape=}")
    print(f"{len(np.unique(adata.obs[key]))} {key} combinations.")

    return adata
