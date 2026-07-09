"""
web_client/ui_helpers.py — UI formatting helpers (view layer).
"""

# 状态条宽度下约一行等宽字符量（仅展示用，与 CSS 单行省略配合）
PROMPT_STATUS_DISPLAY_MAX_CHARS = 56


def truncate_prompt_one_line(prompt: str, max_chars: int = PROMPT_STATUS_DISPLAY_MAX_CHARS) -> str:
    """压成单行并限制长度，超出用省略号（避免状态 Markdown 换行撑高、频繁重排）。"""
    t = " ".join((prompt or "").split())
    if max_chars <= 0:
        return ""
    if len(t) <= max_chars:
        return t
    if max_chars == 1:
        return "…"
    return t[: max_chars - 1] + "…"


def format_current_prompt_md(prompt: str) -> str:
    """右侧槽位：单行展示当前生效 Prompt（与输入框可不同步时以推理侧为准）。"""
    line = truncate_prompt_one_line(prompt)
    return f"**当前 Prompt**: {line}"


def format_status_md(data: dict) -> str:
    """左侧状态条：仅统计与速度（不含 Prompt）。"""
    return (
        f"已生成: {data['block_count']} blocks | {data['frame_count']} 帧 | {data['elapsed']:.1f}s\n"
        f"推理速度: {data['bps']:.2f} blocks/s | {data['fps']:.1f} fps"
    )


def format_progress_bar_html(played: int, generated: int, total: int) -> str:
    """生成赛博朋克风格双层进度条 HTML。

    已播放 = 品红 (#ff2a6d)
    已生成未播放 = 青色 (#05d9e8)
    未生成 = 深色背景
    """
    if total <= 0:
        return ""

    played_pct = min(100.0, max(0.0, played / total * 100))
    generated_pct = min(100.0, max(0.0, generated / total * 100))

    return (
        f'<div class="fe-progress-wrap">'
        f'<div class="fe-progress-track">'
        f'<div class="fe-progress-generated" style="width: {generated_pct:.1f}%">'
        f'<div class="fe-progress-played" style="width: {(played_pct / generated_pct * 100) if generated_pct > 0 else 0:.1f}%">'
        f'</div></div></div>'
        f'<div class="fe-progress-label">'
        f'<span class="fe-progress-played-label">已播放 {played}/{total} 帧</span>'
        f'<span class="fe-progress-generated-label">已生成 {generated}/{total} 帧</span>'
        f'</div></div>'
    )
