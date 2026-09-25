"""The paper's current vector workflow; original geometry and labels."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

WIDTH, HEIGHT = 3.50, 1.35  # One IEEE column; compact even at full column width.

LAYOUT_HEIGHT = 2.35  # Logical coordinates, separate from printed dimensions.

INK = "#183D4B"

TEAL = "#17617A"

MUTED = "#4E626B"

LINE = "#557581"

PALE = "#F1F7F9"

BORDER = "#95B0BB"

CAPTION = (
    "The CHIA loop. Pale-blue blocks represent simulation and evaluation; "
    "dark blocks represent model-development and round-completion "
    "activities. During development, training inputs and memory-request traces "
    "are inspectable, including oracle, candidate, and baseline observations. "
    "The separate dashed block marks optional synthetic tests and replay. "
    "Both feed statistics and memory-request traces back along the shared "
    "development-feedback arrow. Training and validation "
    "evaluation provide metrics to the independent reviewer, not raw request "
    "traces; the reviewer also receives model sources. Validation metrics are "
    "anonymous. Review, qualification, and summarization are grouped only for "
    "presentation: mandatory integrity checks cannot be waived, and the proposer "
    "writes the final summary after the independent review. The selected model "
    "and summary feed the next round. ChampSim supplies the workload frontend; "
    "synthetic tests and replay use Ramulator directly. Test and transfer results "
    "never feed back into search."
)

def elbow_path(points: list[tuple[float, float]], radius: float = .055) -> MplPath:
    """An orthogonal polyline with small, vector-native rounded corners."""
    vertices = [points[0]]
    codes = [MplPath.MOVETO]
    for before, corner, after in zip(points, points[1:], points[2:]):
        incoming = (corner[0] - before[0], corner[1] - before[1])
        outgoing = (after[0] - corner[0], after[1] - corner[1])
        d_in = (incoming[0] ** 2 + incoming[1] ** 2) ** .5
        d_out = (outgoing[0] ** 2 + outgoing[1] ** 2) ** .5
        r = min(radius, d_in / 2, d_out / 2)
        enter = tuple(corner[j] - r * incoming[j] / d_in for j in (0, 1))
        leave = tuple(corner[j] + r * outgoing[j] / d_out for j in (0, 1))
        vertices.extend([enter, corner, leave])
        codes.extend([MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3])
    vertices.append(points[-1])
    codes.append(MplPath.LINETO)
    return MplPath(vertices, codes)

def draw() -> tuple[plt.Figure, list[tuple[object, tuple[float, ...]]]]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "chia-loop-workflow-v10", "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, WIDTH), ylim=(0, LAYOUT_HEIGHT))
    ax.axis("off")
    fitted = []

    def label(x, y, text, *, size=7.0, color=INK, weight="normal", ha="center", **kw):
        return ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                       ha=ha, va="center", linespacing=1.25, zorder=4, **kw)

    def card(x, y, w, h, lines, *, fill="white", edge=LINE, color=INK, dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
            facecolor=fill, edgecolor=edge, linewidth=.65,
            linestyle=(0, (3, 2)) if dashed else "solid", zorder=2,
        ))
        for offset, text, size, weight in lines:
            artist = label(x + w / 2, y + offset, text, size=size, color=color, weight=weight)
            fitted.append((artist, (x, y, w, h)))

    def arrow(points, *, color=LINE, width=.75):
        ax.add_patch(FancyArrowPatch(
            path=elbow_path(points, radius=.04), arrowstyle="-|>", mutation_scale=6.0,
            linewidth=width, color=color, capstyle="round", joinstyle="round", zorder=1,
        ))

    # All simulation/evaluation nodes share one fill. Synthetic/replay tools use
    # Ramulator directly, so this color does not assert a common CPU frontend.
    card(1.96, 1.73, 1.45, .34, [
        (.17, "Training evaluation", 7.4, "bold"),
    ], fill=PALE, edge=BORDER)
    card(1.96, 1.25, 1.45, .44, [
        (.325, "Synthetic / replay", 7.4, "bold"),
        (.112, "(optional)", 7.0, "normal"),
    ], fill=PALE, edge=BORDER, dashed=True)
    # Put the tools beside Develop: a short fork replaces the tall diagnostic row.
    ax.plot([1.74, 1.83], [1.60, 1.60], color=TEAL, linewidth=.75,
            solid_capstyle="round", zorder=1)
    ax.plot([1.83, 1.83], [1.44, 1.90], color=TEAL, linewidth=.75,
            solid_capstyle="round", zorder=1)
    for y in (1.90, 1.44):
        arrow([(1.83, y), (1.96, y)], color=TEAL)
    # These are outputs, not another operation: join both tool outputs and label
    # the shared arrow that returns them to the developing agent.
    for y in (1.90, 1.44):
        ax.plot([3.41, 3.46], [y, y], color=TEAL, linewidth=.75,
                solid_capstyle="round", zorder=1)
    ax.plot([3.46, 3.46], [1.44, 1.90], color=TEAL, linewidth=.75,
            solid_capstyle="round", zorder=1)
    arrow([(3.46, 1.90), (3.46, 2.23), (1.15, 2.23), (1.15, 1.87)], color=TEAL)
    label(2.305, 2.23, "Statistics + request traces", size=7.2, color=TEAL,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": .8})

    card(.34, 1.37, 1.40, .50, [
        (.35, "Develop", 8.5, "bold"),
        (.12, "Model + summaries", 7.0, "normal"),
    ], fill=TEAL, edge=TEAL, color="white")
    arrow([(.02, 1.60), (.34, 1.60)])
    label(.18, 1.75, "Start", size=7.0, color=MUTED)
    card(.34, .60, 1.40, .50, [
        (.35, "Training", 8.2, "bold"),
        (.12, "Full visibility", 7.2, "normal"),
    ], fill=PALE, edge=BORDER)
    card(.34, .06, 1.40, .50, [
        (.35, "Validation", 8.2, "bold"),
        (.12, "Anonymous", 7.2, "normal"),
    ], fill=PALE, edge=BORDER)

    # Full training visibility belongs to the proposer during development. The
    # independent review receives projected metrics and model sources, not the
    # underlying memory-request traces. Do not suggest otherwise on these edges.
    arrow([(.75, 1.37), (.75, 1.22), (.12, 1.22), (.12, .29), (.34, .29)])
    arrow([(.12, .83), (.34, .83)])
    label(.435, 1.22, "candidate", size=7.0, color=MUTED,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
    card(2.23, .06, 1.18, .94, [
        (.47, "Review,\nQualify,\nSummarize", 8.0, "bold"),
    ], fill=TEAL, edge=TEAL, color="white")
    for y in (.83, .29):
        arrow([(1.74, y), (2.23, y)])
        label(1.985, y + .105, "metrics", size=7.0, color=MUTED)
    arrow([(2.82, 1.00), (2.82, 1.125), (1.50, 1.125), (1.50, 1.37)])
    label(2.16, 1.125, "next round", size=7.0, color=MUTED,
          bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
    return fig, fitted

def validate_layout(fig, fitted) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, (x, y, w, h) in fitted:
        bounds = artist.get_window_extent(renderer).transformed(fig.axes[0].transData.inverted())
        if not (x + .035 <= bounds.x0 and bounds.x1 <= x + w - .035
                and y + .02 <= bounds.y0 and bounds.y1 <= y + h - .02):
            raise ValueError(f"Text does not fit its card: {artist.get_text()!r}")
    for artist in fig.axes[0].texts:
        bounds = artist.get_window_extent(renderer)
        if not (bounds.x0 >= 0 and bounds.y0 >= 0
                and bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height):
            raise ValueError(f"Text extends outside figure: {artist.get_text()!r}")
    return {"card_labels_checked": len(fitted), "text_clipping": False}
