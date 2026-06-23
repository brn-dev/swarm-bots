from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoder, MATQCCDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoder, MATQCSDecoderSelfAttentionMode


OUTPUT_DIR = Path(__file__).resolve().parent
N_AGENTS = 5

BACKGROUND = (248, 250, 252, 255)
PANEL_BACKGROUND = (255, 255, 255, 255)
GRID_LINE = (203, 213, 225, 255)
TEXT = (15, 23, 42, 255)
MUTED_TEXT = (71, 85, 105, 255)
ALLOWED = (34, 197, 94, 255)
BLOCKED = (226, 232, 240, 255)
SAFETY_ONLY = (245, 158, 11, 255)


@dataclass(frozen=True)
class MaskPanel:
    title: str
    row_labels: list[str]
    column_labels: list[str]
    allowed: torch.Tensor
    safety_only: torch.Tensor | None = None
    note: str | None = None


def main() -> None:
    panels = build_mask_panels(n_agents=N_AGENTS)
    for panel in panels:
        image = draw_mask_panel(panel)
        image.save(OUTPUT_DIR / f"{slugify(panel.title)}.png")

    overview = draw_overview(panels)
    overview.save(OUTPUT_DIR / "mat_attention_masks.png")


def build_mask_panels(*, n_agents: int) -> list[MaskPanel]:
    return [
        build_qcc_query_context_panel(n_agents=n_agents),
        build_qcc_context_self_panel(n_agents=n_agents),
        build_qcx_query_context_panel(n_agents=n_agents),
        build_qcs_panel(
            n_agents=n_agents,
            mode=MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY,
            title="QCS context_tokens_only self-attention",
        ),
        build_qcs_panel(
            n_agents=n_agents,
            mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
            title="QCS full_causal self-attention",
        ),
    ]


def build_qcc_query_context_panel(*, n_agents: int) -> MaskPanel:
    decoder = make_qcc_decoder(n_agents=n_agents)
    mask, has_visible_context, _key_padding_mask = decoder._build_parallel_context_attention_mask(
        batch_size=1,
        n_queries=n_agents,
        n_contexts=n_agents,
        context_mask=None,
        device=torch.device("cpu"),
    )
    if mask is None or has_visible_context is None:
        raise RuntimeError("QCC query-context mask unexpectedly missing")

    visible_with_safety = ~mask
    effective_allowed = visible_with_safety & has_visible_context[0].unsqueeze(1)
    safety_only = visible_with_safety & ~effective_allowed
    return MaskPanel(
        title="QCC query-to-context attention",
        row_labels=agent_labels(prefix="q", n_agents=n_agents),
        column_labels=agent_labels(prefix="c", n_agents=n_agents),
        allowed=effective_allowed,
        safety_only=optional_mask(safety_only),
        note="Orange marks the internal safety key for rows with no real visible context; output is zeroed.",
    )


def build_qcc_context_self_panel(*, n_agents: int) -> MaskPanel:
    decoder = make_qcc_decoder(n_agents=n_agents)
    mask, has_visible_context, _key_padding_mask = decoder._build_context_self_attention_mask(
        batch_size=1,
        n_contexts=n_agents,
        context_mask=None,
        device=torch.device("cpu"),
    )
    if mask is None or has_visible_context is None:
        raise RuntimeError("QCC context self-attention mask unexpectedly missing")

    visible_with_safety = ~mask
    effective_allowed = visible_with_safety & has_visible_context[0].unsqueeze(1)
    safety_only = visible_with_safety & ~effective_allowed
    return MaskPanel(
        title="QCC context-to-context attention",
        row_labels=agent_labels(prefix="c", n_agents=n_agents),
        column_labels=agent_labels(prefix="c", n_agents=n_agents),
        allowed=effective_allowed,
        safety_only=optional_mask(safety_only),
    )


def build_qcx_query_context_panel(*, n_agents: int) -> MaskPanel:
    decoder = make_qcx_decoder(n_agents=n_agents)
    mask, has_visible_context, _key_padding_mask = decoder._build_parallel_context_attention_mask(
        batch_size=1,
        n_queries=n_agents,
        n_contexts=n_agents,
        context_mask=None,
        device=torch.device("cpu"),
    )
    if mask is None or has_visible_context is None:
        raise RuntimeError("QCX query-context mask unexpectedly missing")

    visible_with_safety = ~mask
    effective_allowed = visible_with_safety & has_visible_context[0].unsqueeze(1)
    safety_only = visible_with_safety & ~effective_allowed
    return MaskPanel(
        title="QCX query-to-context attention",
        row_labels=agent_labels(prefix="q", n_agents=n_agents),
        column_labels=agent_labels(prefix="x", n_agents=n_agents),
        allowed=effective_allowed,
        safety_only=optional_mask(safety_only),
        note="x tokens are action-conditioned context tokens; orange is the safety key for the first query.",
    )


def build_qcs_panel(
        *,
        n_agents: int,
        mode: MATQCSDecoderSelfAttentionMode,
        title: str,
) -> MaskPanel:
    mask = MATQCSDecoder._build_parallel_attention_mask(
        max_agents=n_agents,
        self_attention_mode=mode,
    )
    return MaskPanel(
        title=title,
        row_labels=interleaved_agent_labels(n_agents=n_agents),
        column_labels=interleaved_agent_labels(n_agents=n_agents),
        allowed=~mask,
    )


def make_qcc_decoder(*, n_agents: int) -> MATQCCDecoder:
    return MATQCCDecoder(
        MATQCCDecoderConfig(d_model=8, nhead=1, assume_agent_mask_is_active_prefix=True),
        max_agents=n_agents,
        memory_d_model=8,
    )


def make_qcx_decoder(*, n_agents: int) -> MATQCXDecoder:
    return MATQCXDecoder(
        MATQCXDecoderConfig(d_model=8, nhead=1, assume_agent_mask_is_active_prefix=True),
        max_agents=n_agents,
        input_d_model=8,
        action_d_model=8,
        memory_d_model=8,
    )


def draw_overview(panels: list[MaskPanel]) -> Image.Image:
    rendered_panels = [draw_mask_panel(panel) for panel in panels]
    gutter = 24
    title_height = 58
    width = max(image.width for image in rendered_panels)
    height = title_height + sum(image.height for image in rendered_panels) + gutter * (len(rendered_panels) - 1)
    image = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(size=24, bold=True)
    draw.text((20, 16), "MAT decoder attention masks", font=title_font, fill=TEXT)

    y = title_height
    for panel_image in rendered_panels:
        image.alpha_composite(panel_image, ((width - panel_image.width) // 2, y))
        y += panel_image.height + gutter
    return image


def draw_mask_panel(panel: MaskPanel) -> Image.Image:
    cell_size = 42
    left_margin = 94
    top_margin = 92
    right_margin = 24
    bottom_margin = 68 if panel.note else 36
    title_height = 34
    width = max(left_margin + len(panel.column_labels) * cell_size + right_margin, 640)
    height = top_margin + len(panel.row_labels) * cell_size + bottom_margin + title_height

    image = Image.new("RGBA", (width, height), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(size=18, bold=True)
    label_font = load_font(size=14, bold=False)
    note_font = load_font(size=12, bold=False)

    draw.text((16, 16), panel.title, font=title_font, fill=TEXT)
    draw.text((left_margin, 52), "source token", font=label_font, fill=MUTED_TEXT)
    draw.text((16, top_margin + 2), "target", font=label_font, fill=MUTED_TEXT)

    grid_x = left_margin
    grid_y = top_margin
    for column_idx, label in enumerate(panel.column_labels):
        x = grid_x + column_idx * cell_size + cell_size // 2
        draw_centered_text(draw, (x, grid_y - 20), label, font=label_font, fill=TEXT)

    for row_idx, label in enumerate(panel.row_labels):
        y = grid_y + row_idx * cell_size + cell_size // 2
        draw_centered_text(draw, (grid_x - 26, y), label, font=label_font, fill=TEXT)

    safety_only = panel.safety_only
    for row_idx in range(len(panel.row_labels)):
        for column_idx in range(len(panel.column_labels)):
            x0 = grid_x + column_idx * cell_size
            y0 = grid_y + row_idx * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            fill = BLOCKED
            if bool(panel.allowed[row_idx, column_idx]):
                fill = ALLOWED
            elif safety_only is not None and bool(safety_only[row_idx, column_idx]):
                fill = SAFETY_ONLY
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=GRID_LINE)

    legend_y = grid_y + len(panel.row_labels) * cell_size + 18
    draw_legend_item(draw, (left_margin, legend_y), ALLOWED, "allowed", font=note_font)
    draw_legend_item(draw, (left_margin + 112, legend_y), BLOCKED, "blocked", font=note_font)
    if panel.safety_only is not None:
        draw_legend_item(draw, (left_margin + 224, legend_y), SAFETY_ONLY, "safety only", font=note_font)

    if panel.note is not None:
        draw.text((left_margin, legend_y + 28), panel.note, font=note_font, fill=MUTED_TEXT)

    draw.rectangle((0, 0, width - 1, height - 1), outline=(226, 232, 240, 255))
    return image


def draw_legend_item(
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        color: tuple[int, int, int, int],
        label: str,
        *,
        font: ImageFont.ImageFont,
) -> None:
    x, y = position
    draw.rectangle((x, y, x + 14, y + 14), fill=color, outline=GRID_LINE)
    draw.text((x + 20, y - 1), label, font=font, fill=MUTED_TEXT)


def draw_centered_text(
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        text: str,
        *,
        font: ImageFont.ImageFont,
        fill: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    x = center[0] - width // 2
    y = center[1] - height // 2
    draw.text((x, y), text, font=font, fill=fill)


def agent_labels(*, prefix: str, n_agents: int) -> list[str]:
    return [f"{prefix}{agent_idx}" for agent_idx in range(n_agents)]


def interleaved_agent_labels(*, n_agents: int) -> list[str]:
    labels: list[str] = []
    for agent_idx in range(n_agents):
        labels.extend((f"q{agent_idx}", f"c{agent_idx}"))
    return labels


def optional_mask(mask: torch.Tensor) -> torch.Tensor | None:
    return mask if bool(mask.any()) else None


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "")
        .replace("__", "_")
    )


def load_font(*, size: int, bold: bool) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
