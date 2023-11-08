# import  load modules
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def PLOT_TEX_MAPS(target, parameter_maps, title=None, save = False, text_below = None, subtitle="Recovered"):
    plt.rcParams.update({'font.size': 10})
    #dark background
    plt.style.use('dark_background')
    #no grid
    plt.rcParams['axes.grid'] = False
    if parameter_maps.ndim == 2 and parameter_maps.shape[1] == 5:
        WIDTH = int(np.sqrt(parameter_maps.shape[0]))
        HEIGHT = WIDTH
        parameter_maps = parameter_maps.reshape(WIDTH, HEIGHT, 5)
    elif parameter_maps.ndim == 3 and parameter_maps.shape[2] == 5:
        WIDTH, HEIGHT = parameter_maps.shape[:2]
    else:
        raise ValueError("parameter_maps must have a shape of (-1, 5) or (WIDTH, HEIGHT, 5)")

    labels = [subtitle, "Cm", "Ch", "Bm", "Bh", "T"]
    name = f"tex_maps_{title}.png" if title else "tex_maps.png"
    
    plt.close('all')
    num_plots = 6
    figsize = (24, 5)
    fig = plt.figure(figsize=figsize)
    width_ratios = [1]
    for i in range(5):
        width_ratios.append(1)
        width_ratios.append(0.01)


    gs = GridSpec(1, len(width_ratios), width_ratios=width_ratios, wspace=0.3)



    # Display the original/target image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(target)
    ax0.axis('off')
    ax0.set_title(labels[0])

    for i in range(5):
        ax = fig.add_subplot(gs[0, 2*i+1])  # Plotting maps
        im = ax.imshow(parameter_maps[:, :, i], cmap='viridis')
        ax.axis('off')
        ax.set_title(labels[i+1])

        # Determine the position for the color bar axis
        map_pos = ax.get_position()
        # Adjust the color bar width here (the third value in the list)
        cbar_width = 0.01  # Set this to your desired width
        cbar_pos = [map_pos.x1, map_pos.y0, cbar_width, map_pos.height]

        # Add colorbar for each parameter map, manually setting the position to match map
        cbar_ax = fig.add_axes(cbar_pos)
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical',pad=0.15)
        #remove outline from colorbar
        cbar.outline.set_visible(False)


    plt.subplots_adjust(wspace=0.4,hspace=0.0)

    plt.suptitle(title, fontsize=16)
    #add text below  plots
    if text_below is not None:
        # plt.figtext(0.5, 0.01, text_below, wrap=True, horizontalalignment='center', fontsize=12)
        #lower!!
        plt.figtext(0.5, 0.01, text_below, wrap=True, horizontalalignment='center',  verticalalignment='bottom', fontsize=10)
        # plt.figtext( arg names )
    plt.savefig(name, dpi=100, transparent=True, bbox_inches='tight')
    plt.show()
