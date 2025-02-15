import os
from concurrent.futures import ThreadPoolExecutor
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk
import tkinter as tk
from tkinter import filedialog
import time

def load_cine_mr_data(folder_path):
    # Read all .IMA files from folder
    file_names = []
    file_names.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.IMA')])
    file_names.sort()  # Ensure files are sorted

    # Function to read a single image
    def read_image(file_name):
        return sitk.ReadImage(file_name)

    # Read images in parallel
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(read_image, file_names))

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

def select_points_interactive_continuous(video, window_size=40, dense=False):
    """
    Allows continuous point selection with a zoomed-in view and crosshair cursor for precise selections.

    Parameters:
    - video: The video tensor in the format [b, t, c, h, w].    
    - zoom: Zoom factor for the zoomed-in subplot.
    - window_size: Size of the window around the cursor to zoom in on.

    Returns:
    - A tensor of selected points (x, y) with shape [num_points, 3], where each point is (0, x, y).
    """
    first_frame = video[0, 0].cpu().numpy()
    if dense:
        h, w = first_frame.shape[1], first_frame.shape[2]
        points = [(0, x, y) for y in range(h) for x in range(w)]
        return torch.tensor(points, dtype=torch.float32)
    else:
        # Convert the first frame to numpy and ensure correct format for display
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

def track_one_folder(img_dir):
    video = load_cine_mr_data(img_dir)
    video = video.cuda()
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/cotracker_stride_4_wind_8.pth'
        )
    ).cuda()

    queries = select_points_interactive_continuous(video, dense=False).cuda()
    filename = time.strftime("%Y%m%d-%H%M%S")
    pred_tracks, pred_visibility = model(video, queries=queries[None])

    rpath = img_dir.split('anonymized/')[-1]
    results_dir = os.path.join('./results', rpath)
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, filename + '_queries.npy'), queries.cpu().numpy())
    np.save(os.path.join(results_dir, filename + '_pred_tracks.npy'), pred_tracks.squeeze(0).cpu().numpy())

    # Get the upper folder of img_dir
    sorted_dir = os.path.join(img_dir.split('raw/')[0], 'sorted')
    os.makedirs(sorted_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=sorted_dir,
        linewidth=0.5,
        mode='cool'
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=filename)
    np.save(os.path.join(sorted_dir,'tracks.npy'), pred_tracks.squeeze(0).cpu().numpy())

def main():
    for i in range(2):
        # if i == 0:
        #     w_range = [1]
        #     p_range = [5,17]
        # else:
        if i == 1:
            w_range = [2,3,4,'5-2']
            p_range = [3,5,7]
            for v in [1]:
                for w in w_range:
                    for p in p_range: 
                        img_dir =f'CORONAL_00{p+1:02d}/'
                        try:
                            track_one_folder(img_dir)
                            print(f'Finished tracking v{v}_w{w}_0{p}')
                        except Exception as e:
                            print(f'Error tracking v{v}_w{w}_0{p}: {e}')


if __name__ == "__main__":
    # v = 10
    # w = 3
    # f = 5
    # img_dir =f'/data/anonymized/pyt/data/v{v}_w{w}_0{f}/'
    main()
