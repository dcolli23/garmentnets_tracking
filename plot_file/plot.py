import pdb

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from plot_file.decor_utils import auto_numpy
from plot_file.plot_utils import update_scene_layout, _pointcloud_trace, _volume_trace


@auto_numpy
def plot_pointcloud(data: np.ndarray, fig=None, downsample=5, colors=None, colorscale=None, row=None,
                    col=None, id=1,
                    view='top', dis=2, show_grid=True, center=True, draw_line=False):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        _pointcloud_trace(data, downsample, colors, colorscale=colorscale, draw_line=draw_line),
        row=row, col=col)
    update_scene_layout(fig, scene_id=id, pts=data, center=center, view=view, dis=dis,
                        show_grid=show_grid)

    return fig


'''
Plot 3D mesh, which is a combination of Mesh3d and Scatter3d.
'''


@auto_numpy
def plot_mesh(v, f, fig=None, view='top', row=None, col=None, id=1, dis=2, opacity=0.6, draw_face=True,
              show_grid=True):
    if fig is None:
        fig = go.Figure()
    fig = fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                  i=f[:, 0], j=f[:, 1], k=f[:, 2], colorscale='RdYlBu',
                                  opacity=opacity), row=row, col=col)
    if draw_face:
        tri_pts = v[f]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_pts:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])
        fig = fig.add_trace(go.Scatter3d(x=Xe,
                                         y=Ye,
                                         z=Ze,
                                         mode='lines'
                                         ), row=row, col=col)
    update_scene_layout(fig, scene_id=id, pts=v, view=view, show_grid=show_grid, dis=dis)
    return fig


'''
Create a segmented point cloud figure.
'''


def plot_seg_fig(data: np.ndarray, labels: np.ndarray, labelmap=None, fig=None, show_grid=True):
    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Colormap.
    labels = labels.astype(int)
    for label in np.unique(labels):
        # pdb.set_trace()
        subset = data[np.where(labels == label)]
        # subset = np.squeeze(subset)
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(_pointcloud_trace(subset, name=legend))
    fig.update_layout(showlegend=True)

    update_scene_layout(fig, pts=data, show_grid=show_grid)

    return fig


@auto_numpy
def plot_pointclouds(pcs, labels=None, fig=None, show_grid=True):
    if labels is None:
        labels = [np.ones(pc.shape[0]) * i for i, pc in enumerate(pcs)]
    else:
        labels = [np.ones(pc.shape[0]) * i for i, pc in enumerate(pcs)]

    labels = np.concatenate(labels)
    pcs = np.concatenate(pcs)
    return plot_seg_fig(pcs, labels, fig=fig, show_grid=show_grid)


@auto_numpy
def plot_volume(volume, fig=None):
    if fig is None:
        fig = go.Figure()
    fig.add_trace(_volume_trace(volume))
    return fig


@auto_numpy
def plot_picker_traj(pick_pos, actions, plot_dim=[2, 1], save_path=None, is_softgym=False):
    picker_trajs = pick_pos.reshape(1, 3) + np.cumsum(actions[:, :3], axis=0)  # (H, 3)
    picker_trajs = np.concatenate([pick_pos.reshape(1, 3), picker_trajs], axis=0)
    # print('init and end ', picker_trajs[0], picker_trajs[18:22])
    picker_trajs = picker_trajs[:, plot_dim]
    if is_softgym:
        picker_trajs[:, 0] = -picker_trajs[:, 0]
    if len(plot_dim) == 2:
        plt.plot(picker_trajs[:, 0], picker_trajs[:, 1])
        plt.scatter(picker_trajs[0, 0], picker_trajs[0, 1], color='red')
        if save_path:
            plt.savefig(save_path)
            # plt.close()
        else:
            plt.show()
        plt.close()
    elif len(plot_dim) == 3:
        plot_pointclouds([picker_trajs, picker_trajs[0:1]]).show()


@auto_numpy
def plot_hinge_from_key_pts(key_pts, fig=None, show_grid=True, return_pts=False):
    """
    First 2 points: axis
    Next 2 points: page
    """
    if fig is None:
        fig = go.Figure()
    if len(key_pts.shape) == 3:
        key_pts = key_pts[0]
    if key_pts.shape[0] == 4:
        vs = np.zeros((6, 3))
        vs[:2] = key_pts[:2]
        for i, page_id in enumerate([2, 3]):
            vs[2 * i + 2] = 2 * key_pts[page_id] - key_pts[0]
            vs[2 * i + 3] = 2 * key_pts[page_id] - key_pts[1]

        f = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [0, 4, 5]])
    elif key_pts.shape[0] == 10:
        vs = key_pts[2:]
        f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        fig = fig.add_trace(go.Scatter3d(x=key_pts[:2, 0],
                                         y=key_pts[:2, 1],
                                         z=key_pts[:2, 2],
                                         mode='lines',
                                         line=dict(width=20),
                                         ))

    plot_mesh(vs, f, fig=fig, show_grid=show_grid, draw_face=False)
    fig.update_layout(showlegend=True)

    update_scene_layout(fig, pts=vs, show_grid=show_grid)
    if return_pts:
        return fig, vs
    return fig


@auto_numpy
def plot_hinge_from_key_pts_compare(key_pts, pred_key_pts=None, pc=None, fig=None, show_grid=True,
                                    data_id=0):
    """
    First 2 points: axis
    Next 2 points: page
    """
    if fig is None:
        fig = go.Figure()
    if pc is not None:
        plot_pointcloud(pc, fig=fig, show_grid=show_grid, downsample=1)
    fig, vs = plot_hinge_from_key_pts(key_pts, fig=fig, show_grid=show_grid, return_pts=True)
    if pred_key_pts is not None:
        fig, vs2 = plot_hinge_from_key_pts(pred_key_pts, fig=fig, show_grid=show_grid, return_pts=True)
        vs = np.concatenate([vs, vs2], axis=0)

    update_scene_layout(fig, pts=vs, show_grid=show_grid)
    fig.update_layout(
        title={
            # 'text': f"Flow EPE: {epe:.3f}     Visibility pred acc: {(gt_vis[valid_pred] == pred_vis).mean():.3f}",
            'text': f"Index {data_id}",
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        }
    )
    return fig


"""
histogram plot 
"""


def hists_plot(xs, labels=None, **kwargs):
    df = pd.DataFrame(dict(
        series=np.concatenate([[l] * len(x) for x, l in zip(xs, labels)]),
        data=np.concatenate(xs)
    ))
    f = px.histogram(df, x="data", color="series", barmode="overlay", histnorm="percent",
                     **kwargs)
    return f


@auto_numpy
def plot_confusion_mat(conf_mat, labels):
    """Plot a confusion matrix using plotly"""
    conf_mat = conf_mat[::-1]
    fig = ff.create_annotated_heatmap(conf_mat, x=labels, y=labels[::-1], colorscale='RdBu')
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 25
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
                            x=0.5,
                            y=1.05,
                            showarrow=False,
                            xanchor='center',
                            yanchor='top',
                            text="Prediction",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
                            x=-0.05,
                            y=0.5,
                            showarrow=False,
                            text="Ground truth",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    fig.update_layout(margin=dict(t=200, l=200))
    fig['data'][0]['showscale'] = True
    return fig
