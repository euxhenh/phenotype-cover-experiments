import json
import os
from pathlib import Path
from scGeneFit.functions import get_markers

import jsbeautifier
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import (SelectFromModel, f_classif,
                                       mutual_info_classif)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from src import FReliefF, GreedyCoverSelector, TopDE, CEMSelector
from tqdm import tqdm


class Dummy:
    def __init__(self, f):
        if isinstance(f, str) or isinstance(f, Path):
            self.feature_importances_ = np.loadtxt(f)
        else:
            self.feature_importances_ = f


def fit_and_dump_GCS(
    *,
    root,
    coverage_list,
    data,
    key,
    extras={},
    max_features_list=None,
    json_filename='report.json',
    write_suffix='',
):
    """Fits a greedy cover model and dumps results.

    Parameters:
    root: str or Path
        Base directory.
    coverage_list: list or ndarray
        List of coverage integers.
    data: AnnData object
    key: str
        Key under data.obs to use of labels.
    extras: dict
        Dictionary with extra info to dump.
    max_features_list: list or None
        Maximum coverage per iteration.
    json_filename: str
        Json filename to store results in. Must not be the full path.
    write_suffix: str
        Suffix to append to base directory.
    """
    basedir = Path(root) / ('GreedyCover' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    # Fit model
    ordered = True
    GCS = GreedyCoverSelector(ordered=ordered, multiplier=None)
    GCS.fit(data.X, data.obs[key])

    for i, coverage in enumerate(tqdm(coverage_list)):
        if max_features_list is not None:
            max_cov = max_features_list[i]
        else:
            max_cov = None
        GCS.select(coverage=coverage, max_iters=max_cov)

        select_and_dump(
            GCS,
            dict_key_list=coverage,
            json_path=basedir / json_filename,
            extras={
                **extras,
                'n_features': GCS.n_outputs_,
                'ordered': ordered,
                'n_pairs_no_cover': GCS.n_pairs_with_incomplete_cover_,
                'solution': GCS.solution_indices_ordered_,
                'n_elements_remaining': GCS.n_elements_remaining_,
                'coverage_until': GCS.coverage_until_,
            },
            threshold=0.5,  # Last selected feature has score of 1
            no_sol=True,
        )


def fit_and_dump_DT(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    feature_importances_filename="feature_importances_.txt",
    write_suffix='',
    **kwargs,
):
    """Fits a decision tree model and dumps results.

    Parameters: (See Greedy Cover for those missing here).
    max_features_list: list or ndarray
        Maximum number of features to select per run.
    check_if_fitted: bool
        If True will check the FI directory to see if there are feature
        importanes. If so, will load those instead of fitting a new model.
    feature_importances_filename: str
        Name of FI file.
    kwargs: dict
        Parameter to pass to DecisionTreeClassifier
    """
    basedir = Path(root) / ('DT' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    fi_path = basedir / feature_importances_filename
    if not fi_path.is_file() or not check_if_fitted:
        print("Fitting decision tree.")
        dtc = DecisionTreeClassifier(**kwargs)
        dtc.fit(data.X, data.obs[key])
        np.savetxt(fi_path, dtc.feature_importances_)

    select_and_dump(
        fi_path,
        max_features_list=max_features_list,
        dict_key_list=coverage_list,
        json_path=basedir / json_filename,
        extras=extras,
    )


def fit_and_dump_topDE(
    *,
    root,
    topk_list=list(range(1, 22)),
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    de_dir="DEResults",
    write_suffix='',
    max_features_list=None,
    coverage_list=None,
):
    """Find top DE genes by taking top k genes for every class.
    coverage_list is ignored.
    """
    basedir = Path(root) / ('TopDE' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    fi_path = basedir / de_dir
    if not fi_path.is_dir() or not check_if_fitted:
        print("Finding DE genes.")
        tde = TopDE(verbose=False)
        tde.fit(data.X, data.obs[key])
        tde.dump(fi_path)
    else:
        tde = TopDE(verbose=False)
        tde.load(fi_path)

    max_feats = np.max(max_features_list)
    for topk in tqdm(topk_list):
        tde.select(topk)
        select_and_dump(
            tde,
            dict_key_list=topk,
            json_path=basedir / json_filename,
            extras=extras,
            threshold=0.5,  # Last selected feature has score of 1
        )
        if tde.n_outputs_ > max_feats:
            break


def fit_and_dump_ReliefF(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    feature_importances_filename="feature_importances_.txt",
    write_suffix='',
):
    """Fits and stores the results of ReliefF.
    """
    basedir = Path(root) / ('ReliefF' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    fi_path = basedir / feature_importances_filename
    if not fi_path.is_file() or not check_if_fitted:
        print("Fitting ReliefF.")
        rel = FReliefF(n_neighbors=30, verbose=True)
        rel.fit(data.X, data.obs[key])
        np.savetxt(fi_path, rel.feature_importances_)

    select_and_dump(
        fi_path,
        max_features_list=max_features_list,
        dict_key_list=coverage_list,
        json_path=basedir / json_filename,
        extras=extras,
    )


def fit_and_dump_scGeneFit(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    feature_importances_filename="feature_importances_.txt",
    write_suffix='',
):
    """Fits and stores the results of scGeneFit.
    """
    basedir = Path(root) / ('scGeneFit' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    print("Fitting scGeneFit.")
    for cov, max_feats in zip(coverage_list, max_features_list):
        solution = get_markers(data.X, data.obs[key], max_feats)

        dump(
            features=solution,
            dict_key=cov,
            json_path=basedir / json_filename,
            extras={
                **extras,
                'n_features': len(solution),
            }
        )


def fit_and_dump_Fval(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    extras={},
    json_filename="report.json",
    write_suffix='',
):
    """Computes F values for every feature and stores results.
    """
    basedir = root / ('Fval' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    scores, pvals = f_classif(data.X, data.obs[key])
    dummy = Dummy(scores)

    select_and_dump(
        dummy,
        max_features_list=max_features_list,
        dict_key_list=coverage_list,
        json_path=basedir / json_filename,
        extras=extras,
    )


def fit_and_dump_MI(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    feature_importances_filename="feature_importances_.txt",
    write_suffix='',
):
    """Computes mutual information of features and stores results.
    """
    basedir = Path(root) / ('MI' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    fi_path = basedir / feature_importances_filename
    if not fi_path.is_file() or not check_if_fitted:
        print("Fitting MI.")
        scores = mutual_info_classif(data.X, data.obs[key])
        np.savetxt(fi_path, scores)

    select_and_dump(
        fi_path,
        max_features_list=max_features_list,
        dict_key_list=coverage_list,
        json_path=basedir / json_filename,
        extras=extras,
    )


def fit_and_dump_mRMR(
    *,
    root,
    coverage_list,
    max_features_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    feature_importances_filename="feature_importances_.txt",
    write_suffix='',
):
    """Find most relevant and least redundant set of features and stores.
    """
    basedir = Path(root) / ('mRMR' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    fi_path = basedir / feature_importances_filename
    if not fi_path.is_file() or not check_if_fitted:
        print("Fitting mRMR.")
        le = LabelEncoder()
        yy = le.fit_transform(data.obs[key])
        K = np.max(max_features_list) + 1

        selected_features = mrmr_classif(pd.DataFrame(data.X), pd.Series(yy), K=K)
        feature_importances_ = np.zeros((data.shape[1]))
        N = len(selected_features)
        feature_importances_[selected_features] = np.arange(N+1, 1, -1)
        np.savetxt(fi_path, feature_importances_)

    select_and_dump(
        fi_path,
        max_features_list=max_features_list,
        dict_key_list=coverage_list,
        json_path=basedir / json_filename,
        extras=extras,
        threshold=0.5,
    )


def fit_and_dump_CEM(
    *,
    root,
    coverage_list,
    data=None,
    key=None,
    check_if_fitted=True,
    extras={},
    json_filename="report.json",
    write_suffix='',
    alpha=None,
    rho=None,
    max_features_list=None,
    smoothing_parameter=0.7,
    rs=1000,
):
    """Set cover using Cross Entropy Method.
    """
    basedir = Path(root) / ('CrossEntropy' + write_suffix)
    os.makedirs(basedir, exist_ok=True)

    ordered = True
    CEM = CEMSelector(ordered=ordered, rs=rs, smoothing_parameter=smoothing_parameter)
    CEM.fit(data.X, data.obs[key])

    for coverage in tqdm(coverage_list):
        if alpha is None:
            CEM.alpha = float(coverage) / data.shape[1]
        else:
            CEM.alpha = alpha
        if rho is None:
            CEM.rho = 2 / (5 + float(coverage) // 3)
        else:
            CEM.rho = rho

        CEM.select(coverage=float(coverage))
        select_and_dump(
            CEM,
            dict_key_list=coverage,
            json_path=basedir / json_filename,
            extras={
                **extras,
                'ordered': ordered,
                'rho': CEM.rho,
                'alpha': CEM.alpha,
                'rs': rs,
                'min_coverage': CEM.min_coverage,
                'solution': CEM.ordered_features_,
            },
            threshold=0.5,  # Last selected feature has score of 1
            no_sol=True,
        )


def dump(
    *,
    feature_selector=None,
    features=None,
    dict_key=None,
    json_path=None,
    extras=None,
    no_sol=False,
):
    """Dump results of feature selection to a json.

    Parameters
    __________
    feature_selector: sklearn.feature_selection.SelectFromModel object
        The feature selection method, must implement .get_support()
    features: list
        The features to dump. Will be ignored if feature_selector is None.
    dict_key: str
        What key to use for this dump in the json file
    json_path: str
        Path to an existing json file or path to a json file to be
        created.
    extras: dict
        Dictionary of extra keys to dump along with selected features.
    no_sol: bool
        If True, will not dump the solution, but only the extras.
    """
    if feature_selector is None and features is None:
        raise ValueError(
            "One of `feature_selector` or `features` "
            "must not be None."
        )
    if feature_selector is not None:
        solution = feature_selector.get_support(indices=True)
    else:
        solution = features

    # Temporary report if one not provided
    if json_path is None:
        json_path = 'temp_report.json'

    # Load report if already exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            report = json.load(f)
    else:
        report = {}

    # Determine if there should be a key or not
    if dict_key is None:
        report_to_use = report
    else:
        report[dict_key] = {}
        report_to_use = report[dict_key]

    # Dump any extras
    if extras is not None:
        for extra_key in extras:
            _temp = extras[extra_key]
            if isinstance(_temp, np.ndarray):  # json cannot dump ndarrays
                _temp = _temp.tolist()
            report_to_use[extra_key] = _temp

    if not no_sol:
        if isinstance(solution, np.ndarray):
            solution = solution.tolist()
        report_to_use['solution'] = solution

    # Beautify report
    options = jsbeautifier.default_options()
    options.indent_size = 4
    beau_report = jsbeautifier.beautify(json.dumps(report), options)

    with open(json_path, "w") as f:
        f.write(beau_report)


def get_feat_list(path_to_json):
    """
    Reads a report json file and loads its keys
    and number of features selected.
    """
    with open(path_to_json, "r") as f:
        __report_all_gc_temp = json.load(f)
    coverage_list = list(__report_all_gc_temp.keys())
    n_feats_list = [__report_all_gc_temp[coverage]['n_features']
                    for coverage in coverage_list]
    print(n_feats_list)
    return coverage_list, n_feats_list


def select_and_dump(
    scorer,
    *,
    dict_key_list,
    max_features_list=None,
    json_path=None,
    extras=None,
    threshold=-np.inf,
    no_sol=False,
):
    """Select features using scoring object and dump to a json.

    scorer: object
        Must have a feature_importances_ property
    dict_key_list: list of str
        What key to use for this dump in the json file for each run.
    max_features_list: list of int
        Maximum number of features to select for each run.
    json_path: str
        Path to an existing json file or path to a json file to be
        created.
    extras: dict
        Dictionary of extra keys to dump along with selected features.
    threshold: float
        Threshold to use for selecting features
    """
    if json_path is not None:
        json_path = Path(json_path)
        if not json_path.name.endswith('.json'):
            raise ValueError("Invalid json filename passed.")
        # Make directories if they don't exist
        parent_dir = json_path.parents[0]
        parent_dir.mkdir(parents=True, exist_ok=True)

    # load a dummy with feature importances if passed string
    if isinstance(scorer, str) or isinstance(scorer, Path):
        scorer = Dummy(scorer)

    if not isinstance(dict_key_list, (list, np.ndarray)):
        dict_key_list = [dict_key_list]
    if max_features_list is None:
        max_features_list = [None] * len(dict_key_list)
    elif not isinstance(max_features_list, (list, np.ndarray)):
        max_features_list = [max_features_list]
    if len(max_features_list) != len(dict_key_list):
        raise ValueError(
            "Inconsistent length for `max_features_list` and `dict_key_list`."
        )

    for max_features, dict_key in zip(max_features_list, dict_key_list):
        SFM = SelectFromModel(
            scorer,
            threshold=threshold,
            max_features=max_features,
            prefit=True)

        n_feats = len(SFM.get_support(indices=True))
        if max_features is not None and n_feats < max_features:
            raise ValueError("Found less features than max.")

        dump(
            feature_selector=SFM,
            dict_key=dict_key,
            json_path=json_path,
            extras={
                **extras,
                'n_features': n_feats,
            },
            no_sol=no_sol,
        )
