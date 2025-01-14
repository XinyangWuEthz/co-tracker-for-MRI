import os

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk
import tkinter as tk
from tkinter import filedialog
import time

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory(initialdir='/nfs/asmfsfs03/cptMRgPT/data/anonymized/data_verticalMR/Torso_lowMR', title='Select the data folder')
    return folder_path


def load_cine_mr_data(folder_path):
    # Read all .DCM files
    file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.IMA')]
    file_names.sort()  # Ensure files are sorted

    # Read images and store them in a list
    images = [sitk.ReadImage(f) for f in file_names]

    # Ensure all images have the same size
    size = images[0].GetSize()
    if not all(image.GetSize() == size for image in images):
        raise RuntimeError("Not all images have the same size")

    # Convert SimpleITK images to numpy arrays
    arrays = [sitk.GetArrayFromImage(image) for image in images]

    # Expand channel dimension (from 1 to 3)
    arrays = [np.stack([arr]*3, axis=-1) for arr in arrays]

    # Convert the list to a numpy array
    data = np.stack(arrays, axis=0)
    # Convert from numpy.uint16 to numpy.float32
    data = data.astype(np.float32)

    # Standardize data to RGB range
    data /= data.max()
    data *= 255.0

    # Convert to Torch Tensor and reshape to [b, t, c, h, w]
    tensor = torch.from_numpy(data).float()
    tensor = tensor.permute(1, 0, 4, 2, 3)

    return tensor

# data dir
# root = '/nfs/asmfsfs03/cptMRgPT/data/anonymized'
# rpath = 'data_verticalMR/Torso_lowMR/VMRI04/D1_upright/7_HASTE_dyn_sag_single'

# folder_path = os.path.join(root, rpath)

folder_path = select_folder()
rpath = folder_path.split('anonymized/')[-1]

video = load_cine_mr_data(folder_path)
video = video.cuda()
model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './checkpoints/cotracker_stride_4_wind_8.pth'
    )
).cuda()


def select_points_interactive_continuous(video, window_size=40):
    """
    Allows continuous point selection with a zoomed-in view and crosshair cursor for precise selections.

    Parameters:
    - video: The video tensor in the format [b, t, c, h, w].    
    - zoom: Zoom factor for the zoomed-in subplot.
    - window_size: Size of the window around the cursor to zoom in on.

    Returns:
    - A tensor of selected points (x, y) with shape [num_points, 3], where each point is (0, x, y).
    """
    # Convert the first frame to numpy and ensure correct format for display
    first_frame = video[0, 0].cpu().numpy()
    first_frame = np.transpose(first_frame, (1, 2, 0)).astype(np.uint8)

    points = []
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 5))
    ax[0].imshow(first_frame)
    ax[1].axis('off')

    zoom_ax = fig.add_axes([0.7, 0.7, 0.2, 0.2], anchor='NE', zorder=1)
    zoom_ax.axis('off')

    def update_zoom(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        zoom_ax.clear()
        zoom_ax.axis('off')

        x0, x1 = max(x - window_size // 2, 0), min(x + window_size // 2, first_frame.shape[1])
        y0, y1 = max(y - window_size // 2, 0), min(y + window_size // 2, first_frame.shape[0])
        zoom_img = first_frame[y0:y1, x0:x1]
        
        zoom_ax.imshow(zoom_img, aspect='equal')
        for p in points:
            if x0 <= p[1] <= x1 and y0 <= p[2] <= y1:
                zoom_ax.plot((p[1]-x0), (p[2]-y0), 'ro')

        zoom_ax.axvline(x - x0, color='red')
        zoom_ax.axhline(y - y0, color='red')
        plt.draw()

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((0, int(event.xdata), int(event.ydata)))
            ax[0].plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    def onkey(event):
        if event.key == 'z' and points:
            points.pop()
            ax[0].clear()
            ax[0].imshow(first_frame)
            for point in points:
                ax[0].plot(point[1], point[2], 'ro')
            fig.canvas.draw()
        elif event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('motion_notify_event', update_zoom)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()

    return torch.tensor(points, dtype=torch.float32)

queries = select_points_interactive_continuous(video).cuda()
filename = time.strftime("%Y%m%d-%H%M%S")
pred_tracks, pred_visibility = model(video, queries=queries[None])

os.makedirs(os.path.join('./results', rpath), exist_ok=True)
np.save(os.path.join('./results', rpath, filename + '_queries.npy'), queries.cpu().numpy())
np.save(os.path.join('./results', rpath, filename + '_pred_tracks.npy'), pred_tracks.cpu().numpy())

vis = Visualizer(
    save_dir=os.path.join('./results', rpath),
    linewidth=0.5,
    mode='cool'
)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename=filename)
