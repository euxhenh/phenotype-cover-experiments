import numpy as np
import json
from .plots import line_plot


def key_of_key_list(report, *keys):
    """Constructs a list of the form
    [report[i][key] for i in report] for every key.
    """
    to_return = []
    for key in keys:
        to_return.append(np.asarray([report[i][key] for i in report]))
    return tuple(to_return)


def correct_for_minmax(n_features, *arrs, vmin=-np.inf, vmax=np.inf):
    """Filters out values in n_features that are less than min_n_feats
    or larger than max_n_feats. Also filter the corresponding entires in
    each arr.
    """
    for arr in arrs: assert len(n_features) == len(arr)
    mask = np.ma.masked_inside(n_features, vmin, vmax).mask
    return (
        np.asarray(n_features)[mask],
        *[np.asarray(arr)[mask] for arr in arrs]
    )


def nearest_sol(report, n_features, verbose=False):
    """Assumes n_features is an increasing function of cov."""
    for cov in report:
        if report[cov]['n_features'] == n_features:
            if verbose:
                print(f"Searched {n_features}, found {n_features}")
            return report[cov]['solution']
        if report[cov]['n_features'] > n_features:
            if verbose:
                print(f"Searched {n_features}, found {report[cov]['n_features']}")
            return np.random.choice(
                report[cov]['solution'], n_features, replace=False)
    raise Exception("Not enough features.")


def line_report_metric(
        reps, x_key, y_key, *,
        xlabel=None, ylabel=None,
        vmin=-np.inf, vmax=np.inf,
        imdir=None, im_suffix='',
        **kwargs,
    ):
    """Line plot for metric across all reps.
    Will read reps[method][c][x_key] = x
    and reps[method][c][y_key] = y
    """
    xlist, ylist = [], []
    for rep in reps:
        x_vals, y_vals = key_of_key_list(rep, x_key, y_key)
        x_vals, y_vals = correct_for_minmax(
            x_vals, y_vals, vmin=vmin, vmax=vmax)
        xlist.append(x_vals)
        ylist.append(y_vals)

    return line_plot(
        xlist, ylist,
        xlabel=xlabel if xlabel is not None else x_key,
        ylabel=ylabel if ylabel is not None else y_key,
        savepath=imdir / 'png' / f'{y_key}{im_suffix}.png' if imdir else None,
        savepath_svg=imdir / 'svg' / f'{y_key}{im_suffix}.svg' if imdir else None,
        **kwargs,
    )


def load_reports(root, *, lreports=False, llogics=False, ldeconvs=False, method_list=[]):
    report_paths = [root / m / "report.json" for m in method_list]
    lr_paths = [root / m / "logisticR.json" for m in method_list]
    deconv_paths = [root / m / "deconv.json" for m in method_list]

    reports, logics, deconvs = [], [], []
    to_return = []
    if lreports:
        for report_path in report_paths:
            with open(report_path, "r") as f:
                reports.append(json.load(f))
        to_return.append(reports)
    if llogics:
        for lr_path in lr_paths:
            with open(lr_path, "r") as f:
                logics.append(json.load(f))
        to_return.append(logics)
    if ldeconvs:
        for deconv_path in deconv_paths:
            with open(deconv_path, "r") as f:
                deconvs.append(json.load(f))
        to_return.append(deconvs)
    to_return = tuple(to_return)
    if len(to_return) == 1:
        return to_return[0]
    return to_return
