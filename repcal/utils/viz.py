from repcal.utils.data import to_sentences
from rich.table import Table
from rich.text import Text

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

COLORS = """
#96dde1
#5bd3e4
#00c7ed
#00bbf8
#00adff
#009eff
#008dff
#007aff
#0064ff
#0046ff
#0600ff""".split()


def get_table_compare(original, generated, row=None):
    table = Table(
        show_lines=True,
        # width=1
    )

    table.add_column("Original")
    table.add_column("Generated")

    ori_sentences = to_sentences(original)
    gen_sentences = to_sentences(generated)

    r_embeddings = torch.tensor(model.encode(ori_sentences))
    h_embeddings = torch.tensor(model.encode(gen_sentences))

    sim = torch.mm(r_embeddings,h_embeddings.T)

    gen = ""
    ori = ""

    for i, line in enumerate((ori_sentences)):
        val = sim[i].max().item()
        color = COLORS[int(val * 10)]
        style = "green" if val < 0.9 else ""
        ori = Text.assemble(ori, (f"{val:.2}\t ", color), (line, style), "\n")

    for j, line in enumerate((gen_sentences)):
        val = sim[:,j].max().item()
        color = COLORS[int(val * 10)]
        style = "red" if val < 0.9 else ""
        gen = Text.assemble(gen, (f"{val:.2}\t ", color), (line, style), "\n")

    table_row = [ori, gen]
    table.add_row(*table_row)

def show_sim(p_sens, h_sens, sim, p_label=None, h_label=None):
    if p_label is None:
        p_label = ""
    if h_label is None:
        h_label = ""

    fig, ax = plt.subplots(figsize=(8, 8))  # type: ignore
    ax: plt.Axes
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)

    # We want to show all ticks...
    ax.xaxis.set_ticks(np.arange(len(p_sens)))
    ax.yaxis.set_ticks(np.arange(len(h_sens)))
    # ... and label them with the respective list entries
    ax.xaxis.set_ticklabels(p_sens, fontsize=10)
    ax.yaxis.set_ticklabels(h_sens, fontsize=10)
    ax.grid(False)
    plt.xlabel(f"Premise {p_label}", fontsize=14)
    plt.ylabel(f"Hypothesis {h_label}", fontsize=14)
    title = "Similarity Matrix"
    plt.title(title, fontsize=14)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.xaxis.get_ticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    for i in range(len(h_sens)):
        for j in range(len(p_sens)):
            # if i>=j:continue
            text = ax.text(
                j,
                i,
                "{:.3f}".format(sim[i, j].item()),
                ha="center",
                va="center",
                color="k" if sim[i, j].item() < 0.5 else "w",
            )

    fig.tight_layout()
    plt.show()

def show_tensor(tensor):
    flatten = tensor.view(-1)
    fig, ax = plt.subplots()  # type: ignore
    ax: plt.Axes
    ax.imshow(flatten.view(1, -1), vmin=0, vmax=1, cmap="Blues")

    for j in range(len(flatten)):
        text = ax.text(
            j,
            0,
            "{:.3f}".format(flatten[j].item()),
            ha="center",
            va="center",
            color="k" if flatten[j].item() < 0.5 else "w",
        )

    ax.get_yaxis().set_visible(False)
    plt.show()
