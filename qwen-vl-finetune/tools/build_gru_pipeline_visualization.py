#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEGREE_PATTERN = re.compile(r"(\d+)\s*degree", re.IGNORECASE)
CM_PATTERN = re.compile(r"(\d+)\s*cm", re.IGNORECASE)
FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)

STOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {
    STOP: "stop",
    FORWARD: "forward",
    LEFT: "turn_left",
    RIGHT: "turn_right",
}


def parse_video_id(video_id: str) -> Tuple[str, int]:
    if not isinstance(video_id, str) or "-" not in video_id:
        return str(video_id), 0
    traj, step = video_id.rsplit("-", 1)
    try:
        return traj, int(step)
    except ValueError:
        return traj, 0


def action_codes_from_answer(answer: str) -> List[int]:
    answer = (answer or "").lower()

    if "right" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        return [RIGHT] * max(1, steps)

    if "left" in answer:
        match = DEGREE_PATTERN.search(answer)
        steps = int(match.group(1)) // 15 if match else 1
        return [LEFT] * max(1, steps)

    if "move forward" in answer or "forward" in answer:
        match = CM_PATTERN.search(answer)
        steps = int(match.group(1)) // 25 if match else 1
        return [FORWARD] * max(1, steps)

    if "stop" in answer:
        return [STOP]

    return []


def extract_frame_index(frame_id: str) -> int:
    m = FRAME_PATTERN.search(str(frame_id))
    return int(m.group(1)) if m else -1


def load_annotations(annotation_path: Path) -> List[dict]:
    if not annotation_path.exists():
        return []
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_traj_steps(annotations: List[dict], traj: str) -> List[Tuple[int, List[int], str]]:
    rows: List[Tuple[int, List[int], str]] = []
    prefix = f"{traj}-"
    for ann in annotations:
        vid = str(ann.get("video_id", ""))
        if not vid.startswith(prefix):
            continue
        _, step = parse_video_id(vid)
        ans = str(ann.get("a", ""))
        rows.append((step, action_codes_from_answer(ans), ans))
    rows.sort(key=lambda x: x[0])
    return rows


def cumulative_map(step_rows: List[Tuple[int, List[int], str]]) -> Dict[int, List[int]]:
    acc: List[int] = []
    out: Dict[int, List[int]] = {}
    for step, codes, _ in step_rows:
        acc.extend(codes)
        out[step] = list(acc)
    return out


def cumulative_until_inclusive(cmap: Dict[int, List[int]], step_inclusive: int) -> List[int]:
    use_step = None
    for step in sorted(cmap.keys()):
        if step <= step_inclusive:
            use_step = step
        else:
            break
    if use_step is None:
        return []
    return cmap.get(use_step, [])


def action_histogram(codes: List[int]) -> Dict[str, int]:
    stats = {"stop": 0, "forward": 0, "turn_left": 0, "turn_right": 0}
    for c in codes:
        name = ACTION_NAMES.get(int(c), "unknown")
        if name in stats:
            stats[name] += 1
    return stats


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def round_num(v: Any, digits: int = 3) -> Any:
    if isinstance(v, float):
        return round(v, digits)
    return v


def compact_vectors(seq: List[List[float]], max_steps: int = 4, digits: int = 3) -> str:
    if not isinstance(seq, list) or len(seq) == 0:
        return "[]"
    if len(seq) <= max_steps:
        shown = seq
        compact = [[round_num(x, digits) for x in row] for row in shown]
        return json.dumps(compact, ensure_ascii=False)

    shown = seq[: max_steps - 1] + [seq[-1]]
    compact = [[round_num(x, digits) for x in row] for row in shown]
    body = json.dumps(compact, ensure_ascii=False)
    return body[:-1] + ', "...", [last step]]'


def compact_array(arr: List[float], max_items: int = 12, digits: int = 4) -> str:
    if not isinstance(arr, list):
        return "[]"
    vals = [round_num(x, digits) for x in arr[:max_items]]
    out = json.dumps(vals, ensure_ascii=False)
    if len(arr) > max_items:
        out = out[:-1] + ', "..."]'
    return out


def render_structured_messages(messages: List[dict]) -> str:
    blocks: List[str] = []
    for i, msg in enumerate(messages):
        role = html_escape(msg.get("role", "unknown"))
        content = msg.get("content", [])
        items_html: List[str] = []
        for item in content:
            t = item.get("type", "")
            if t == "text":
                txt = html_escape(str(item.get("text", ""))).replace("\n", "<br/>")
                items_html.append(f"<div class='msgtext'>{txt}</div>")
            elif t == "image":
                img = html_escape(str(item.get("image", "")))
                items_html.append(f"<div class='msgimg'>image: {img}</div>")
            else:
                items_html.append(f"<div class='msgtext'>{html_escape(item)}</div>")
        blocks.append(
            f"<div class='msgcard'><div class='msghead'>Message {i+1} - role={role}</div>{''.join(items_html)}</div>"
        )
    return "\n".join(blocks)


def build_html(payload: dict, annotation_path: Path, slot_rows: List[dict], step_rows: List[Tuple[int, List[int], str]]) -> str:
    source = payload.get("raw_source_0", {})
    q = source.get("q", "")
    a = source.get("a", "")
    messages = payload.get("raw_messages_0", [])
    diag = payload.get("gru_motion_diagnostics", {})

    shapes = {
        "gru_input_shape": diag.get("gru_input_shape"),
        "gru_hidden_shape": diag.get("gru_hidden_shape"),
        "projected_shape": diag.get("projected_shape"),
        "logits_shape": payload.get("logits_shape"),
    }

    action_rows_html = "\n".join(
        f"<tr><td>{step}</td><td>{html_escape(ans)}</td><td>{html_escape(codes)}</td><td>{len(codes)}</td></tr>"
        for step, codes, ans in step_rows[:80]
    )

    slot_rows_html = "\n".join(
        f"<tr><td>{r['slot']}</td><td>{html_escape(r['frame_id'])}</td><td>{r['frame_idx']}</td><td>{r['step_target']}</td><td>{r['gru_length']}</td><td>{r['prefix_len_before_fallback']}</td><td>{html_escape(r['prefix_preview'])}</td></tr>"
        for r in slot_rows
    )

    motion_positions = payload.get("motion_positions_0", [])
    hist = action_histogram([c for r in slot_rows for c in r["prefix_codes"]])

    max_gru = max((r["gru_length"] for r in slot_rows), default=1)
    bars = "\n".join(
        f"<div class='barrow'><span class='label'>H{r['slot']}</span><div class='bar'><div class='fill' style='width:{(r['gru_length']/max_gru)*100:.1f}%;'></div></div><span class='val'>{r['gru_length']}</span></div>"
        for r in slot_rows
    )

    gru_inputs = diag.get("gru_input_per_slot_0", [])
    motion_outputs = diag.get("motion_output_per_slot_0", [])
    slot_io_rows: List[str] = []
    for i, r in enumerate(slot_rows):
        in_seq = gru_inputs[i] if i < len(gru_inputs) else []
        out_vec = motion_outputs[i] if i < len(motion_outputs) else []
        slot_io_rows.append(
            "<tr>"
            f"<td>H{r['slot']}</td>"
            f"<td>{r['gru_length']}</td>"
            f"<td><code>{html_escape(compact_vectors(in_seq, max_steps=4, digits=3))}</code></td>"
            f"<td><code>{html_escape(compact_array(out_vec, max_items=12, digits=4))}</code></td>"
            "</tr>"
        )
    slot_io_html = "\n".join(slot_io_rows)
    messages_html = render_structured_messages(messages)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>GRU Motion Pipeline Visualizer</title>
<style>
:root {{
  --bg:#f7f6f2;
  --ink:#1f2937;
  --muted:#6b7280;
  --card:#ffffff;
  --line:#e5e7eb;
  --brand:#0f766e;
  --brand2:#f59e0b;
  --brand3:#3b82f6;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:'IBM Plex Sans', 'Segoe UI', sans-serif; color:var(--ink); background:radial-gradient(circle at 8% 0%, #fff5db 0, transparent 35%), radial-gradient(circle at 90% 10%, #dff7f2 0, transparent 32%), var(--bg); }}
.wrap {{ max-width:1280px; margin:0 auto; padding:24px; }}
.hero {{ background:linear-gradient(130deg, #0f766e 0%, #115e59 60%, #134e4a 100%); color:#ecfeff; border-radius:18px; padding:24px; box-shadow:0 20px 45px rgba(0,0,0,0.18); }}
.hero h1 {{ margin:0 0 10px; font-size:30px; }}
.hero .meta {{ display:flex; flex-wrap:wrap; gap:10px; }}
.chip {{ border:1px solid rgba(236,254,255,.35); border-radius:999px; padding:6px 12px; font-size:13px; }}
.grid {{ display:grid; grid-template-columns:repeat(12,1fr); gap:14px; margin-top:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 8px 20px rgba(0,0,0,0.06); }}
.c12 {{ grid-column:span 12; }}
.c8 {{ grid-column:span 8; }}
.c6 {{ grid-column:span 6; }}
.c4 {{ grid-column:span 4; }}
h2 {{ margin:0 0 10px; font-size:18px; }}
small, .muted {{ color:var(--muted); }}
.pipeline {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; align-items:stretch; }}
.stage {{ border:1px dashed var(--line); border-radius:12px; padding:10px; background:#fafafa; }}
.stage h3 {{ margin:0 0 6px; font-size:14px; color:#111827; }}
.stage p {{ margin:0; font-size:12px; color:var(--muted); }}
.table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.table th,.table td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
.table th {{ background:#f8fafc; position:sticky; top:0; }}
pre {{ margin:0; background:#0b1020; color:#dbeafe; border-radius:12px; padding:12px; overflow:auto; max-height:280px; font-size:12px; }}
code {{ background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:4px 6px; font-size:12px; display:block; white-space:pre-wrap; word-break:break-word; }}
.kpi {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
.k {{ background:#f9fafb; border:1px solid var(--line); border-radius:12px; padding:10px; }}
.k .v {{ font-weight:700; margin-top:6px; }}
.barrow {{ display:grid; grid-template-columns:40px 1fr 50px; gap:10px; align-items:center; margin-bottom:8px; }}
.bar {{ height:10px; border-radius:999px; background:#e5e7eb; overflow:hidden; }}
.fill {{ height:100%; background:linear-gradient(90deg, var(--brand3), var(--brand)); }}
.msgcard {{ border:1px solid var(--line); border-radius:12px; margin-bottom:10px; overflow:hidden; }}
.msghead {{ background:#f8fafc; font-weight:600; padding:8px 10px; font-size:13px; }}
.msgtext,.msgimg {{ padding:8px 10px; border-top:1px solid var(--line); font-size:13px; line-height:1.45; }}
.msgimg {{ color:#334155; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
@media (max-width:980px) {{
  .c8,.c6,.c4 {{ grid-column:span 12; }}
  .kpi {{ grid-template-columns:repeat(2,1fr); }}
  .pipeline {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class=\"wrap\">
  <section class=\"hero\">
    <h1>GRU -> Motion Token -> LLM Pipeline Visualizer</h1>
    <div class=\"meta\">
      <span class=\"chip\">video_id: {html_escape(source.get('video_id', 'N/A'))}</span>
      <span class=\"chip\">motion_token_id: {html_escape(payload.get('motion_token_id', 'N/A'))}</span>
      <span class=\"chip\">loss: {html_escape(payload.get('loss', 'N/A'))}</span>
      <span class=\"chip\">annotation_db: {html_escape(str(annotation_path))}</span>
    </div>
  </section>

  <div class=\"grid\">
    <section class=\"card c12\">
      <h2>End-to-End Flow</h2>
      <div class=\"pipeline\">
        <div class=\"stage\"><h3>1) Annotation DB</h3><p>Read trajectory actions from annotations.json grouped by video_id step.</p></div>
        <div class=\"stage\"><h3>2) Frame Slots</h3><p>Select 8 frames and compute per-slot frame cutoffs.</p></div>
        <div class=\"stage\"><h3>3) GRU Prefix Build</h3><p>Build prefix action sequence per slot and convert to 7D motion features.</p></div>
        <div class=\"stage\"><h3>4) GRU + Projector</h3><p>Encode with GRU, pick valid slot output, project to 4096D motion token embeddings.</p></div>
        <div class=\"stage\"><h3>5) LLM Fusion</h3><p>Insert projected motion tokens at &lt;motion&gt; positions in prompt embeddings.</p></div>
      </div>
    </section>

    <section class=\"card c12\">
      <h2>Shapes and Token Placement</h2>
      <div class=\"kpi\">
        <div class=\"k\"><small>gru_input_shape</small><div class=\"v\">{html_escape(shapes['gru_input_shape'])}</div></div>
        <div class=\"k\"><small>gru_hidden_shape</small><div class=\"v\">{html_escape(shapes['gru_hidden_shape'])}</div></div>
        <div class=\"k\"><small>projected_shape</small><div class=\"v\">{html_escape(shapes['projected_shape'])}</div></div>
        <div class=\"k\"><small>logits_shape</small><div class=\"v\">{html_escape(shapes['logits_shape'])}</div></div>
      </div>
      <div style=\"margin-top:10px;\"><small>motion_positions_0: {html_escape(motion_positions)}</small></div>
    </section>

    <section class=\"card c6\">
      <h2>Action Set Extracted From DB</h2>
      <small>Parsed from answers in annotations for this trajectory.</small>
      <table class=\"table\">
        <thead><tr><th>Step</th><th>Answer</th><th>Action Codes</th><th>Len</th></tr></thead>
        <tbody>{action_rows_html}</tbody>
      </table>
    </section>

        <section class="card c12">
            <h2>GRU Input And Motion Token Output (Per 8 Slots)</h2>
            <small>GRU input vectors are truncated to first 3 steps + last step and rounded to 3 decimals. Motion token output shows first 12 dims of 4096.</small>
            <table class="table">
                <thead><tr><th>Slot</th><th>GRU Len</th><th>GRU Input (7D vectors)</th><th>Projected Motion Token (4096D preview)</th></tr></thead>
                <tbody>{slot_io_html}</tbody>
            </table>
        </section>

    <section class=\"card c6\">
      <h2>Slot -> Frame -> GRU Input Mapping</h2>
      <small>What each History slot actually feeds into GRU.</small>
      <table class=\"table\">
        <thead><tr><th>Slot</th><th>Frame</th><th>Frame Idx</th><th>Step Target</th><th>GRU Len</th><th>Raw Prefix Len</th><th>Prefix Preview</th></tr></thead>
        <tbody>{slot_rows_html}</tbody>
      </table>
    </section>

    <section class=\"card c4\">
      <h2>GRU Length Profile</h2>
      {bars}
    </section>

    <section class=\"card c4\">
      <h2>Aggregated Action Mix</h2>
      <div class=\"kpi\" style=\"grid-template-columns:1fr;\">
        <div class=\"k\"><small>forward</small><div class=\"v\">{hist['forward']}</div></div>
        <div class=\"k\"><small>turn_left</small><div class=\"v\">{hist['turn_left']}</div></div>
        <div class=\"k\"><small>turn_right</small><div class=\"v\">{hist['turn_right']}</div></div>
        <div class=\"k\"><small>stop</small><div class=\"v\">{hist['stop']}</div></div>
      </div>
    </section>

    <section class=\"card c4\">
      <h2>Instruction + Target</h2>
      <div><small>q</small></div>
      <pre>{html_escape(q)}</pre>
      <div style=\"margin-top:8px;\"><small>a</small></div>
      <pre>{html_escape(a)}</pre>
    </section>

        <section class="card c12">
            <h2>Prompt Structured View (Messages To LLM)</h2>
            <small>Same content as raw prompt, grouped by role and content blocks.</small>
            {messages_html}
        </section>
  </div>
</div>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rich GRU pipeline visualization HTML from debug JSON.")
    parser.add_argument("--debug_json", required=True, help="Path to debug/gru_forward_exact_*.json")
    parser.add_argument("--output_html", default="", help="Output HTML path")
    parser.add_argument("--annotation_path", default="", help="Optional explicit annotations.json path")
    args = parser.parse_args()

    debug_json = Path(args.debug_json)
    payload = json.loads(debug_json.read_text(encoding="utf-8"))

    source = payload.get("raw_source_0", {})
    video_id = str(source.get("video_id", ""))
    traj, _ = parse_video_id(video_id)

    data_path = Path(str(source.get("data_path", "")))
    inferred_ann = data_path.parent / "annotations.json" if data_path else Path("")
    annotation_path = Path(args.annotation_path) if args.annotation_path else inferred_ann

    annotations = load_annotations(annotation_path)
    step_rows = build_traj_steps(annotations, traj)
    cmap = cumulative_map(step_rows)

    frame_ids: List[str] = []
    for msg in payload.get("raw_messages_0", []):
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text":
                text = str(item.get("text", ""))
                if "Frame id:" in text:
                    frame_ids.append(text.split("Frame id:", 1)[1].split("\\n", 1)[0].strip())

    frame_step_targets = payload.get("frame_step_targets_0", [])
    gru_lengths = payload.get("gru_lengths", [[[]]])
    if isinstance(gru_lengths, list) and len(gru_lengths) > 0 and isinstance(gru_lengths[0], list):
        slot_lengths = gru_lengths[0]
    else:
        slot_lengths = []

    slots = min(len(frame_step_targets), len(slot_lengths), len(frame_ids))
    slot_rows: List[dict] = []
    for i in range(slots):
        step_target = int(frame_step_targets[i])
        raw_prefix = cumulative_until_inclusive(cmap, step_target - 1)
        max_prefix_len = max(1, step_target - 1)
        capped = raw_prefix[:max_prefix_len]
        prefix_codes = capped if capped else [STOP]
        slot_rows.append(
            {
                "slot": i + 1,
                "frame_id": frame_ids[i],
                "frame_idx": extract_frame_index(frame_ids[i]),
                "step_target": step_target,
                "gru_length": int(slot_lengths[i]),
                "prefix_len_before_fallback": len(raw_prefix),
                "prefix_codes": prefix_codes,
                "prefix_preview": prefix_codes[:20],
            }
        )

    output_html = Path(args.output_html) if args.output_html else debug_json.with_name("gru_pipeline_visualization.html")
    output_html.write_text(build_html(payload, annotation_path, slot_rows, step_rows), encoding="utf-8")
    print(f"Wrote rich pipeline visualization: {output_html}")


if __name__ == "__main__":
    main()
