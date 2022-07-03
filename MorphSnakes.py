import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import morphsnakes as ms

PATH_IMG_NODULE = 'Dataset\\US0023_0171.png'


def visual_callback_2d(background, fig=None):
    """
        Returns a callback than can be passed as the argument `iter_callback`
        of `morphological_geodesic_active_contour` and
        `morphological_chan_vese` for visualizing the evolution
        of the levelsets. Only works for 2D images.

        Parameters
        ----------
        background : (M, N) array
            Image to be plotted as the background of the visual evolution.
        fig : matplotlib.figure.Figure
            Figure where results will be drawn. If not given, a new figure
            will be created.

        Returns
        -------
        callback : Python function
            A function that receives a levelset and updates the current plot
            accordingly. This can be passed as the `iter_callback` argument of
            `morphological_geodesic_active_contour` and
            `morphological_chan_vese`.

        """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def example_nodule():

        # Load the image.
        img = imread(PATH_IMG_NODULE)[..., 0] / 255.0

        # g(I)
        gimg = ms.inverse_gaussian_gradient(img, alpha=1000, sigma=5.48)

        # Initialization of the level-set.
        init_ls = ms.circle_level_set(img.shape, (130, 170), 20)
        # init_ls = ms.circle_level_set(img.shape, (170, 130), 20)
        # Callback for visual plotting
        callback = visual_callback_2d(img)

        # MorphGAC.
        ms.morphological_geodesic_active_contour(gimg, iterations=36,
                                                 init_level_set=init_ls,
                                                 smoothing=10, threshold=0.4,
                                                 balloon=1, iter_callback=callback)


if __name__ == '__main__':
    example_nodule()
    plt.show()
