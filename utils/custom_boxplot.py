import numpy as np
from matplotlib import pyplot as plt
from seaborn.categorical import _BoxPlotter
from seaborn.utils import remove_na


class _CustomBoxPlotter(_BoxPlotter):
    def draw_boxplot(self, ax, kws):
        """Use matplotlib to draw a boxplot on an Axes."""
        vert = self.orient == "v"

        props = {}
        for obj in ["box", "whisker", "cap", "median", "flier"]:
            props[obj] = kws.pop(obj + "props", {})

        max_lens = []
        # calculate max num of levels
        for i, group_data in enumerate(self.plot_data):
            max_len = 0
            for j, hue_level in enumerate(self.hue_names):
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = np.asarray(remove_na(group_data[hue_mask]))
                if box_data.size == 0:
                    continue

                max_len += 1

            max_lens.append(max_len)
        max_levels = np.max(max_lens)


        for i, group_data in enumerate(self.plot_data):
            # Draw nested groups of boxes
            # n_levels = len(self.hue_names)
            # each_width = self.width / (n_levels - 1)
            # offsets = np.linspace(0, self.width - each_width, n_levels - 1)
            # offsets -= offsets.mean()
            # offsets = np.append(offsets, 0.0)
            # widths = [self.width / len(self.hue_names) * .98 for _ in range(n_levels)]
            # # widths.append(self.width * 0.98)

            box_datas = []
            colors = []

            for j, hue_level in enumerate(self.hue_names):

                # Add a legend for this hue level
                if not i:
                    self.add_legend_data(ax, self.colors[j], hue_level)

                # Handle case where there is data at this level
                if group_data.size == 0:
                    continue

                hue_mask = self.plot_hues[i] == hue_level
                box_data = np.asarray(remove_na(group_data[hue_mask]))

                # Handle case where there is no non-null data
                if box_data.size == 0:
                    continue

                box_datas.append(box_data)
                colors.append(j)

            n_levels = len(box_datas)
            each_width = self.width / (n_levels)
            offsets = np.linspace(0, self.width - each_width, n_levels)
            offsets -= offsets.mean()
            offsets = np.append(offsets, 0.0)
            widths = [self.width / max_levels * .9 * self.width for _ in range(n_levels)]
            # widths = [self.width / n_levels * .9 for _ in range(n_levels)]
            # widths.append(self.width * 0.98)

            for j in range(len(box_datas)):
                center = i + offsets[j]
                artist_dict = ax.boxplot(box_datas[j],
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[center],
                                         widths=widths[j],
                                         **kws)
                self.restyle_boxplot(artist_dict, self.colors[colors[j]], props)
                # Add legend data, but just for one set of boxes
    def restyle_boxplot(self, artist_dict, color, props):
        """Take a drawn matplotlib boxplot and make it look nice."""
        for box in artist_dict["boxes"]:
            box.update(dict(facecolor=tuple(0.7*x + 0.3 for x in color),
                            zorder=.9,
                            edgecolor=color,
                            linewidth=self.linewidth))
            box.update(props["box"])
        for whisk in artist_dict["whiskers"]:
            whisk.update(dict(color=color,
                              linewidth=self.linewidth,
                              linestyle="-"))
            whisk.update(props["whisker"])
        for cap in artist_dict["caps"]:
            cap.update(dict(color=color,
                            linewidth=self.linewidth))
            cap.update(props["cap"])
        for med in artist_dict["medians"]:
            med.update(dict(color=color,
                            zorder=1.0,
                            linewidth=self.linewidth))
            med.update(props["median"])
        for fly in artist_dict["fliers"]:
            fly.update(dict(markerfacecolor=color,
                            marker="d",
                            markeredgecolor=color,
                            markersize=self.fliersize))
            fly.update(props["flier"])


def custom_dodge_boxplot(
    *,
    x=None, y=None,
    hue=None, data=None,
    order=None, hue_order=None,
    orient=None, color=None, palette=None, saturation=.75,
    width=.8, dodge=True, fliersize=5, linewidth=None,
    whis=1.5, ax=None,
    **kwargs
):

    plotter = _CustomBoxPlotter(x, y, hue, data, order, hue_order,
                                    orient, color, palette, saturation,
                                    width, dodge, fliersize, linewidth)

    if ax is None:
        ax = plt.gca()
    kwargs.update(dict(whis=whis))

    plotter.plot(ax, kwargs)
    return ax