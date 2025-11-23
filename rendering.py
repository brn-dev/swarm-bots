import matplotlib
from IPython.display import HTML, display
from matplotlib import animation, pyplot as plt

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def display_image(image, *, cmap=None, dpi=70):
    """Render a single numpy-backed image inline inside a notebook."""
    if image.ndim not in (2, 3):
        raise ValueError("Only 2D grayscale or 3D color images are supported.")

    height, width = image.shape[:2]
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    ax.imshow(image, cmap=cmap)
    display(fig)
    plt.close(fig)

