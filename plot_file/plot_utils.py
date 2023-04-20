import io
import os
import pdb
from typing import Dict, Any

import cv2
import numpy as np
import tqdm
from PIL import Image
from joblib import delayed
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from plotly import graph_objects as go

def get_img_from_fig(fig, dpi=180, width=300, height=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    return img

def write_text_to_img(img, text, pos=None, scale=1, color=(0, 0, 255)):  # Inplace modification
    img = np.ascontiguousarray(img, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = img.shape
    fontScale = scale
    fontColor = color
    thickness = 2

    # dynamically changing the size of the text, so it won't go out of the boundary
    textSize = cv2.getTextSize(text, fontFace=font, fontScale=fontScale, thickness=thickness)[0]
    while textSize[0] > w * 0.8:
        fontScale -= 0.1
        textSize = cv2.getTextSize(text, fontFace=font, fontScale=fontScale, thickness=thickness)[0]

    if pos is None:
        bottomLeftCornerOfText = (w//2-textSize[0]//2, 50)
    else:
        bottomLeftCornerOfText = (int(w * pos[0]), int(h * pos[1]))

    if isinstance(text, str) or isinstance(text, int):
        cv2.putText(img, f'{text}',
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness)
    else:
        cv2.putText(img, '{:.2f}'.format(text),
                    bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness)
    return img


def save_numpy_as_gif(array, filename, fps=20, scale=1.0, add_index_rate=1):
    """Creates a gif given a stack of images using moviepy
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    if add_index_rate > 0:
        for i in range(array.shape[0]):
            array[i] = write_text_to_img(array[i], str(i * add_index_rate))
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_gif(filename, fps=fps)
    # return clip


def get_plot_view(view='top', dis=2.):
    view_dict = {
        'top': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=0., y=dis, z=0.)
        ),
        'front': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=0, y=0, z=dis)
        ),
        'side': dict(
            up=dict(x=0, y=2, z=0),
            eye=dict(x=dis, y=0, z=0.)
        ),
        'tilt': dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=dis / 2, y=dis, z=-dis)
        ),
    }
    return view_dict[view]


def make_grid(array, ncol=5, padding=0, pad_value=120, index_img=False):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    if N % ncol > 0:
        res = ncol - N % ncol
        array = np.concatenate([array, np.ones([res, H, W, C])])
        N = array.shape[0]
    nrow = N // ncol
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]],
                     constant_values=pad_value, mode='constant')
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]],
                             constant_values=pad_value, mode='constant')
            if index_img:
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (150, 150)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cur_img = cv2.putText(cur_img, str(idx), org, font, fontScale, color, thickness,
                                      cv2.LINE_AA)
            row = np.hstack([row, cur_img])
        idx += 1
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img.astype(np.int32)


def _pointcloud_trace(points: np.ndarray, downsample=1, colors=None, colorscale=None,
                      draw_line=False, **kwargs) -> go.Scatter3d:
    marker_dict: Dict[str, Any] = {"size": 1.8}
    if draw_line:
        line_dict: Dict[str, Any] = {"color": "red", "width": 2.}
        mode = "lines+markers"
    else:
        line_dict = {}
        mode = "markers"

    if colorscale is not None and colors is not None:
        marker_dict['color'] = colors
        marker_dict['colorscale'] = colorscale
    elif colors is not None:
        if isinstance(colors, str):
            marker_dict['color'] = colors
        else:
            try:
                a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
                marker_dict["color"] = a
            except:
                marker_dict["color"] = colors[::downsample]
    return go.Scatter3d(
        x=points[::downsample, 0],
        y=points[::downsample, 1],
        z=points[::downsample, 2],
        mode=mode,
        marker=marker_dict,
        line=line_dict,
        **kwargs
    )

def _volume_trace(volume):
    # render a volumetric representation
    h, w, z = volume.shape
    X, Y, Z = np.meshgrid(np.arange(h),
                          np.arange(w),
                          np.arange(z))

    # print('?', np.mean(volume[Y.flatten(), X.flatten(), Z.flatten()] == volume.flatten()))

    trace = go.Volume(
        x=Y.flatten(),
        y=X.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        opacity=0.3,
        surface_count=21
    )
    return trace


def update_scene_layout(fig, scene_id=1, pts=None, center=True, view='top', dis=2, show_grid=True):
    """
    Configure the layout of scene for 3d plot
    """
    camera = get_plot_view(view, dis=dis)
    scene_cfg = dict(
        xaxis=dict(nticks=10),
        yaxis=dict(nticks=10),
        zaxis=dict(nticks=10),
        aspectratio=dict(x=1, y=1, z=1),
        camera=camera,
    )
    if pts is not None and center:
        mean = pts.mean(axis=0)
        max_x = np.abs(pts[:, 0] - mean[0]).max()
        max_y = np.abs(pts[:, 1] - mean[1]).max()
        max_z = np.abs(pts[:, 2] - mean[2]).max()
        all_max = max(max(max_x, max_y), max_z)
        for i, axis in enumerate(['xaxis', 'yaxis', 'zaxis']):
            scene_cfg[axis]['range'] = [mean[i] - all_max, mean[i] + all_max]

    if not show_grid:
        for axis in ['xaxis', 'yaxis', 'zaxis']:
            scene_cfg[axis].update(dict(
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"))
            scene_cfg[axis + '_title'] = ''
    fig.update_layout({f'scene{scene_id}': scene_cfg,
                       'legend': dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                       'margin': dict(l=0, r=0, b=0, t=0),
                       })


def rotate_z(x, y, z, theta):
    w = x + 1j * y
    return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


def get_rotating_frames(fig, scene_num=1, frames_num=60, width=600, height=600, parallel=None):
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    step = 6.26 / frames_num

    def rotate_and_save(fig, t):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        for i in range(scene_num):
            fig.update_layout(
                {f'scene{i + 1}_camera_eye': dict(x=xe, y=ye, z=ze)}
            )
        return Image.open(io.BytesIO(fig.to_image(format="png", width=width, height=height)))

    if parallel is None:
        frames = []
        for t in tqdm.tqdm(np.arange(0, 6.26, step)):
            frames.append(rotate_and_save(fig, t))
    else:
        frames = parallel(delayed(rotate_and_save)(fig, t) for t in np.arange(0, 6.26, step))
    return frames


