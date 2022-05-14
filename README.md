To select features, first run all cells in `FeatureSelection.ipynb`.
This will create a folder `DAT` for each dataset under `data/DAT`.
For each random seed x, the results will be saved under `data/DAT/RS/rsx`
in json files named `report.json`.

For classification and deconvolution run all cells in `Classification.ipynb`
and `Deconvolution.ipynb`, respectively. These will save json files with
names `logisticR.json` and `deconv.json` under the same folder as above.

GSEA q-values have been stored under `data/DAT/method_phenotype.json`. To
recompute these, see the notebook `Functional Analysis.ipynb`.

G-PC and CEM-PC algorithms for scRNA-seq data have been implemented as a
separate package <https://github.com/euxhenh/phenotype-cover> and can be installed via
```
pip install phenotype_cover
```
The algorithm for running multiset multicover can be found at <https://github.com/euxhenh/multiset-multicover> and can be installed with
```
pip install multiset-multicover
```

The IPF, HCA, MC datasets can be found from the respective papers. For
convenience, we have made them available for download as `h5ad` files:

* IPF: <https://drive.google.com/file/d/1KDo4IgmI90DEnksReU4uEq9qEl0NEuck/view?usp=sharing>
* HCA: <https://drive.google.com/file/d/1nxnJyOC07e8ST4E9JZKWzsG5uue2laDK/view?usp=sharing>
* MC: <https://drive.google.com/file/d/1Jp2OT0B2prFk_k4ZGr_DysLGwBDAwvui/view?usp=sharing>

Store the files under `data`.

Finally, clone the `RankCorr` repository by running the following at the root of this repo
```
git clone https://github.com/ahsv/RankCorr
```