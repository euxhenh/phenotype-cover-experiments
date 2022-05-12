import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from functools import partial
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import pdist, squareform

PALETTE = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#B3B3B3", "#E5C494", "#9C9DBB", "#E6946B", "#DA8DC4", "#AFCC63",
    "#F2D834", "#E8C785", "#BAB5AE", "#D19C75", "#AC9AAD", "#CD90C5",
    "#B8C173", "#E5D839", "#ECCA77", "#C1B7AA", "#BBA37E", "#BC979D",
    "#C093C6", "#C1B683", "#D8D83E", "#F0CD68", "#C8BAA5", "#A6AB88",
    "#CC958F", "#B396C7", "#CBAB93", "#CCD844", "#F3D05A", "#CFBCA1",
    "#90B291", "#DC9280", "#A699C8", "#D4A0A3", "#BFD849", "#F7D34B",
    "#D6BF9C", "#7BBA9B", "#EC8F71", "#999CC9", "#DD95B3", "#B2D84E",
    "#FBD63D", "#DDC198"
]


def get_col_i(pal, i):
    return pal[i % len(pal)]


def despine_all(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_heatmap(hm, classes=None, exclude_diag=True, savepath=None, logp1=True):
    """
    Plot a heatmap.

    Args:
        hm: square matrix representing the heatmap
        classes: None or array of class labels
    """
    if logp1:
        hm = np.log(hm + 1)

    width = height = 600
    if classes is not None:
        n_classes = len(classes)
        width = height = max(n_classes * 10, 800)
        font_size = max(3, 13 - n_classes // 10)
    zmin = None
    if exclude_diag:
        zmin = np.min(hm[~np.eye(hm.shape[0], dtype=bool)])
    fig = px.imshow(hm, x=classes, y=classes,
                    width=width, height=height, zmin=zmin)
    if classes is not None:
        fig.update_xaxes(tickfont_size=font_size)
        fig.update_yaxes(tickfont_size=font_size)

    if savepath is not None:
        fig.write_image(savepath)
    else:
        fig.show()


def line_plot(
    xlist=None,
    ylist=None,
    *,
    stdlist=None,
    labellist=None,
    zorderlist='reverse',
    title=None,
    xlabel=None,
    ylabel=None,
    palette='qualitative',
    savepath=None,
    savepath_svg=None,
    logy=False,
    legend_loc='lower right',
    ax=None,
):
    """Line plots.

    xlist: list
        list of lists containing x coordinates. If ylist is None,
        this will specify y coordinates instead.
    ylist: list
    labellist: list
    zorderlist: str or array of length len(xlist)
        Order of overlay in the plot.
    title, xlabel, ylabel: str
    palette: str
        One of 'qualitative' or 'gradient'
    savepath: str
        Path to save image to.
    logy: bool
        Whether to plot the y-axis in log scale
    legend_loc: str
        Legend location
    ax: plt axis
    """
    if xlist is None:
        raise ValueError("xlist cannot be None.")

    markers = ['o', '.', 'x', '*', '>', '|', 'X', '1', ',', 'd', 'p', '-']
    linestyles = np.tile(['-', '--', ':', '-.'], 3)
    if palette == 'qualitative':
        pal = list(cm.get_cmap('Dark2').colors)
        pal_buf = cm.get_cmap('tab10').colors
        pal.extend([pal_buf[3], pal_buf[-1]])
    elif palette == 'sequential':
        pal = sns.color_palette("magma", n_colors=len(xlist))
    else:
        pal = sns.color_palette(palette, n_colors=len(xlist))

    if len(xlist) > len(pal):
        raise ValueError("More lines than colors.")

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    if zorderlist == 'reverse':
        zorderlist = np.arange(len(xlist))[::-1]

    for i, x in enumerate(xlist):
        ax.plot(
            x, ylist[i] if ylist is not None else None,
            color=pal[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=(len(x) + 9) // 10,
            zorder=zorderlist[i] if zorderlist is not None else None,
            label=labellist[i] if labellist is not None else None,
        )

        if stdlist is not None:
            # convert to numpy in case these are lists so we can subtract
            # elementwise
            yy = np.asarray(ylist[i])
            sstd = np.asarray(stdlist[i])
            ax.fill_between(x, yy - sstd, yy + sstd, color=pal[i], alpha=0.1)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13)
    if title is not None:
        ax.set_title(title)
    if labellist is not None:
        ax.legend()
    if logy:
        plt.yscale('log')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=5,
        width=2,
        pad=5,
        labelsize=13,
    )
    ax.grid(True, which='major', axis='y', alpha=0.3)
    ax.legend(loc=legend_loc, fontsize='medium')
    # plt.rcParams.update({'font.size': 11})

    #ax.figure.set_size_inches((6, 6))
    if savepath is not None:
        plt.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')
    if savepath_svg is not None:
        plt.savefig(savepath_svg, dpi=300, transparent=True, bbox_inches='tight')


def dotplot(
    X,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    cmap='Blues',
    xticklabels=None,
    yticklabels=None,
    return_fig=False,
    savepath=None,
    colorrow=True,
    savepath_svg=None,
    fontsize=11,
    radius=None,
    ax=None,
    pal=PALETTE,
    imsize=(10, 10),
):
    """Heatmap-like dotplot.
    """
    max_rad = X.max()
    N, M = X.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))

    if ax is None:
        fig, ax = plt.subplots()

    shapes = [
        plt.Circle,
        partial(RegularPolygon, numVertices=3),
        partial(RegularPolygon, numVertices=4)]

    circles = [
        shapes[i%len(shapes)]((i, j),
        color = get_col_i(pal, i) if colorrow else get_col_i(pal, j),
        radius=r / (2 * max_rad) if radius is None else radius)
        for r, i, j in zip(X.flat, x.flat, y.flat)]


#     circles = []
#     for r, i, j in zip(X.flat, x.flat, y.flat):
#         func = shapes[j % len(shapes)] if colorrow else shapes[i % len(shapes)]
#         func = shapes[0]
#         color = get_col_i(pal, i) if colorrow else get_col_i(pal, j)
#         radius=r / (2 * max_rad) if radius is None else radius
#         circles.append(func((i, j), color=color, radius=radius))

    col = PatchCollection(circles, match_original=True)
    ax.add_collection(col)

    if xticklabels is None:
        xticklabels = np.arange(M)
    if yticklabels is None:
        yticklabels = np.arange(N)
    ax.set(xticks=np.arange(M), yticks=np.arange(N),
       xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
    ax.tick_params('x', rotation=90, which='both', bottom=False, left=False)
    ax.tick_params('y', which='both', bottom=False, left=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title)

    ax.figure.set_size_inches(imsize)

    ax.grid(True, which='major', axis='both', alpha=0.3)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')
    if savepath_svg is not None:
        plt.savefig(savepath_svg, dpi=300, transparent=True, bbox_inches='tight')
    if return_fig:
        return ax
    plt.show()


def dendrodot(
    X,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    cmap='Blues',
    xticklabels=None,
    yticklabels=None,
    return_fig=False,
    savepath=None,
    savepath_svg=None,
    fontsize=11,
    radius=None,
    ax=None,
    imsize=(10, 10)
):
    fig, ax = plt.subplots()


def pairwise_distances_heatmap(
    X,
    y=None,
    *,
    title=None,
    xlabel=None,
    ylabel=None,
    cmap='cubehelix',
    savepath=None,
    savepath_svg=None,
    return_fig=False,
    fontsize=13,
    imsize=(6, 6),
):
    """Plot a heatmap of pairwise distances. Order the points by
    y if y is not None.
    """
    p_mat = squareform(pdist(X))

    argidx = np.argsort(y)
    labels, counts = np.unique(y, return_counts=True)
    # Compute mid-way x coordinates
    heatmap_ticks = np.cumsum(counts) - counts // 2

    fig, ax = plt.subplots()

    ax = sns.heatmap(
        p_mat[argidx][:, argidx],
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
        annot=False,
        cbar_kws={'shrink': 0.9},
        square=True,
        ax=ax,
    )

    ax.set_xticks(heatmap_ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(heatmap_ticks)
    ax.set_yticklabels(labels)
    ax.tick_params('x', rotation=90)
    ax.tick_params('both', labelsize=fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    fig.set_size_inches(imsize)
    if savepath is not None:
        plt.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')
    if savepath_svg is not None:
        plt.savefig(savepath_svg, dpi=300, transparent=True, bbox_inches='tight')
    if return_fig:
        return fig, ax
    plt.show()


def scatter(x, y, color):
    color = color.astype(str)
    unq_colors = np.unique(color)
    fig = px.scatter(
        x=x,
        y=y,
        color=color,
        category_orders={'color': unq_colors},
        opacity=0.8,
        labels={'color': 'Cluster ID'},
        color_discrete_map={str(i): get_col_i(PALETTE, i)
                            for i in range(len(unq_colors))},
        render_mode='webgl')

    fig.update_layout(
        showlegend=True,
        plot_bgcolor='#FFFFFF',
        dragmode='lasso',
        xaxis={'visible': False},
        yaxis={'visible': False}
    )

    return fig
