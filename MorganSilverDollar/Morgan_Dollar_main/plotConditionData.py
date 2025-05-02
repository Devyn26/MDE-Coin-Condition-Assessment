import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mplcursors
from matplotlib.patheffects import withSimplePatchShadow
from scipy import stats

df = None


# This block of code essentially provides the ability to hover over data points on a plot to view their respective data
def show_hover_panel(get_text_func=None):
    cursor = mplcursors.cursor(
        hover=2,  # Transient
        annotation_kwargs=dict(
            bbox=dict(
                boxstyle="square,pad=0.5",
                facecolor="white",
                edgecolor="#ddd",
                linewidth=0.5,
                path_effects=[withSimplePatchShadow(offset=(1.5, -1.5))],
            ),
            linespacing=1.5,
            arrowprops=None,
        ),
        highlight=True,
        highlight_kwargs=dict(linewidth=2),
    )

    if get_text_func:
        cursor.connect(
            event="add",
            func=lambda sel: sel.annotation.set_text(get_text_func(sel.index)),
        )

    return cursor


def on_add(index):
    item = df.iloc[index]
    if len(item) == 3:
        parts = [
            f"Grade: {item.Coin}",
            f"Difference: {item.FDensity:,.0f}",
            f"Inventory: #{item.Grade:,.0f}",
        ]
    else:
        parts = [
            f"Coin: {item.Coin}",
            f"FDensity TopR: {item.FDensity_TopR:,.0f}",
            f"FDensity BotR: {item.FDensity_BotR:,.0f}",
            f"FDensity BotL: {item.FDensity_BotL:,.0f}",
            f"FDensity TopL: {item.FDensity_TopL:,.0f}",
            f"DL Grade: ${item.Grade:,.0f}",
        ]

    return "\n".join(parts)


def scatterPlot(x, y, files, title, xlabel, inventory):

    global df
    df = pd.DataFrame(
        dict(
            Coin=files,
            FDensity=x,
            Grade=inventory,
        )
    )

    plt.scatter(x=x, y=y, c=(files/13))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Test Coin ID")
    show_hover_panel(on_add)
    plt.show()
