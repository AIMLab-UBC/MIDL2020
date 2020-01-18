from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import string


def compare_patch_slide(patch_metric, slide_metric, metric='Kappa'):
    plt.style.use('ggplot')
    rc('font', weight='bold')
    colors = ["#bc5090", "#ffa600"]

    x_labels = []
    for idx in range(len(patch_metric)):
        patch_ax = plt.scatter(
            idx, patch_metric[idx], color=colors[0], marker='P', s=90)
        slide_ax = plt.scatter(
            idx, slide_metric[idx], color=colors[1], marker='*', s=90)
        x_labels += [string.ascii_uppercase[idx]]

    plt.xticks(np.arange(len(patch_metric)), x_labels, rotation=65)
    plt.xlabel("Splits")
    plt.ylabel("Kappa")
    plt.legend((patch_ax, slide_ax), ('Patch', 'Slide'), loc=0)
    plt.title('Patch Level vs. Slide Level {}'.format(metric))
    plt.show()
