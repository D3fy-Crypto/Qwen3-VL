"""
generate_qa.py  –  v2
Generates spatial/movement Q&A pairs for every record in r2r_alignment_dataset.json.

25 semantic categories, 6 phrasings each = 150 unique question strings.
Selection is uniform across all VALID categories for a given GRU sequence,
with phrasing variant chosen deterministically from the record id hash.
"""

import json, hashlib, ijson

# ──────────────────────────────────────────────────────────────────────────────
# GRU analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze(gru):
    """Return rich stats dict. Returns None for trivially empty sequences."""
    seq = gru[1:] if gru and gru[0] == 0 else gru[:]
    if not seq:
        return None

    # Run-length encode
    segs = []
    i = 0
    while i < len(seq):
        c, n = seq[i], 1
        while i + n < len(seq) and seq[i + n] == c:
            n += 1
        segs.append((c, n))
        i += n

    fwd   = sum(cnt for c, cnt in segs if c == 1)
    left  = sum(cnt for c, cnt in segs if c == 2)
    right = sum(cnt for c, cnt in segs if c == 3)
    total_move = fwd + left + right

    # first / last non-stop action
    first_code = segs[0][0]  if segs else None
    last_code  = segs[-1][0] if segs else None
    last_n     = segs[-1][1] if segs else 0

    # rotation before first forward step
    rot_pre, rot_pre_dir = 0, None
    for c in gru:
        if c == 1: break
        if c == 2: rot_pre += 15; rot_pre_dir = "left"
        elif c == 3: rot_pre += 15; rot_pre_dir = "right"

    # first forward index in full gru
    first_fwd_idx = next((i for i, c in enumerate(gru) if c == 1), None)

    # alternation switches (consecutive different non-stop codes)
    ns = [c for c, n in segs]
    switches = sum(1 for i in range(1, len(ns)) if ns[i] != ns[i-1])

    # segment counts by type
    fwd_segs   = [(c, n) for c, n in segs if c == 1]
    turn_segs  = [(c, n) for c, n in segs if c in (2, 3)]
    left_segs  = [(c, n) for c, n in segs if c == 2]
    right_segs = [(c, n) for c, n in segs if c == 3]

    # longest run by type
    max_fwd_run   = max((n for c, n in segs if c == 1), default=0)
    max_left_run  = max((n for c, n in segs if c == 2), default=0)
    max_right_run = max((n for c, n in segs if c == 3), default=0)
    max_turn_run  = max(max_left_run, max_right_run)

    # distance / angle after final turn
    fwd_after_last_turn = 0
    for c, n in reversed(segs):
        if c == 1: fwd_after_last_turn += n * 25
        else: break

    # phase strings
    phase_str = []
    for c, n in segs:
        if c == 1:   phase_str.append(f"move forward {n*25}cm")
        elif c == 2: phase_str.append(f"turn left {n*15}°")
        elif c == 3: phase_str.append(f"turn right {n*15}°")

    net_rot = right * 15 - left * 15   # + = net right

    return dict(
        segs=segs, phase_str=phase_str,
        fwd=fwd, left=left, right=right,
        fwd_cm=fwd*25, left_deg=left*15, right_deg=right*15,
        net_rot=net_rot, total_rot=(left+right)*15,
        has_fwd=fwd>0, has_left=left>0, has_right=right>0,
        has_turns=(left+right)>0,
        total_move=total_move,
        first_code=first_code, last_code=last_code, last_n=last_n,
        first_fwd_idx=first_fwd_idx,
        rot_pre=rot_pre, rot_pre_dir=rot_pre_dir,
        switches=switches,
        n_fwd_segs=len(fwd_segs), n_turn_segs=len(turn_segs),
        n_left_segs=len(left_segs), n_right_segs=len(right_segs),
        max_fwd_run=max_fwd_run, max_turn_run=max_turn_run,
        max_left_run=max_left_run, max_right_run=max_right_run,
        fwd_after_last_turn=fwd_after_last_turn,
        n_segs=len(segs),
        seq_len=len(gru),
        fwd_frac=fwd/total_move if total_move else 0,
        ends_fwd=(last_code == 1), starts_fwd=(first_code == 1),
    )


def _d(deg):
    return "right" if deg >= 0 else "left"

# ──────────────────────────────────────────────────────────────────────────────
# 25 Q/A generators  (return (q, a) or None if not applicable)
# ──────────────────────────────────────────────────────────────────────────────

CATS = []
def cat(fn): CATS.append(fn); return fn

# 1 ── total forward distance
@cat
def total_forward(s, v):
    if not s["has_fwd"]: return None
    qs = ["How far does the robot travel forward in total?",
          "What is the total forward distance covered by the robot?",
          "What is the cumulative forward displacement of the robot?",
          "How many centimeters does the robot move forward altogether?",
          "Calculate the total distance the robot travels in the forward direction.",
          "What is the sum of all forward steps in terms of distance?"]
    n, cm = s["fwd"], s["fwd_cm"]
    a = f"The robot moves forward a total of {cm}cm ({n} step{'s' if n>1 else ''} × 25cm)."
    return qs[v % 6], a

# 2 ── total rotation
@cat
def total_rotation(s, v):
    if not s["has_turns"]: return None
    qs = ["What is the total cumulative rotation of the robot?",
          "How many degrees does the robot rotate across the entire sequence?",
          "What is the sum of all rotational steps in degrees?",
          "By how many degrees does the robot turn throughout this sequence?",
          "What is the total angular movement performed by the robot?",
          "How much does the robot rotate in total, counting all turns?"]
    l, r = s["left_deg"], s["right_deg"]
    if l > 0 and r > 0:
        a = f"The robot rotates a total of {l+r}° — {l}° to the left and {r}° to the right."
    elif l > 0:
        a = f"The robot rotates {l}° to the left ({s['left']} steps × 15° each)."
    else:
        a = f"The robot rotates {r}° to the right ({s['right']} steps × 15° each)."
    return qs[v % 6], a

# 3 ── net rotation direction
@cat
def net_rotation_dir(s, v):
    if not s["has_turns"] or s["left_deg"] == s["right_deg"]: return None
    qs = ["What is the robot's net rotation direction?",
          "Does the robot turn more to the left or to the right overall?",
          "What is the predominant turning direction in this sequence?",
          "Which direction does the robot rotate more — left or right?",
          "What is the net angular bias of the robot's motion?",
          "On balance, does the robot rotate towards the left or the right?"]
    net = abs(s["net_rot"]); d = _d(s["net_rot"])
    a = (f"The robot turns predominantly to the {d}, "
         f"with a net rotation of {net}° in that direction "
         f"({s['left_deg']}° left vs {s['right_deg']}° right).")
    return qs[v % 6], a

# 4 ── movement phases
@cat
def movement_phases(s, v):
    if s["n_segs"] < 2: return None
    qs = ["Describe the robot's movement broken down into distinct phases.",
          "What are the key motion phases in this sequence?",
          "Break down the robot's trajectory into its movement segments.",
          "How does the robot's motion unfold step by step?",
          "Outline the sequence of distinct movements the robot performs.",
          "List the individual motion segments that make up this sequence."]
    phases = "; ".join(f"({i+1}) {p}" for i, p in enumerate(s["phase_str"]))
    a = f"The robot moves in {s['n_segs']} phase{'s' if s['n_segs']>1 else ''}: {phases}."
    return qs[v % 6], a

# 5 ── first forward step index
@cat
def first_forward_step(s, v):
    if not s["has_fwd"] or s["first_fwd_idx"] is None or s["first_fwd_idx"] < 2: return None
    qs = ["At which step does the robot first move forward?",
          "When in the sequence does the robot begin moving forward?",
          "How many steps pass before the robot first moves forward?",
          "Which step number marks the robot's first forward movement?",
          "After how many rotation steps does the robot start moving forward?",
          "At what point in the sequence does the robot first translate?"]
    idx = s["first_fwd_idx"]; rot = s["rot_pre"]; d = s["rot_pre_dir"] or "a given"
    a = (f"The robot first moves forward at step {idx}, "
         f"after completing {rot}° of {d} rotation.")
    return qs[v % 6], a

# 6 ── rotation before first forward
@cat
def rotation_before_first_forward(s, v):
    if not s["has_fwd"] or s["rot_pre"] == 0: return None
    qs = ["How much does the robot rotate before taking its first forward step?",
          "What is the initial rotation before the robot begins moving forward?",
          "How does the robot orient itself before it starts translating?",
          "What rotation precedes the first forward movement?",
          "By how many degrees does the robot turn before first moving forward?",
          "How much heading adjustment does the robot make before starting forward motion?"]
    a = f"The robot rotates {s['rot_pre_dir']} {s['rot_pre']}° before taking its first forward step."
    return qs[v % 6], a

# 7 ── last action
@cat
def last_action(s, v):
    if not s["phase_str"]: return None
    qs = ["What is the last action the robot performs in this sequence?",
          "How does the robot's motion sequence end?",
          "What motion concludes this sequence?",
          "Describe the robot's final movement.",
          "What is the terminal action in this motion sequence?",
          "With which movement does the robot finish this sequence?"]
    c, n = s["last_code"], s["last_n"]
    if c == 1:   a = f"The sequence ends with the robot moving forward {n*25}cm."
    elif c == 2: a = f"The sequence ends with the robot turning left {n*15}°."
    else:        a = f"The sequence ends with the robot turning right {n*15}°."
    return qs[v % 6], a

# 8 ── number of direction switches
@cat
def n_switches(s, v):
    if s["switches"] < 2: return None
    qs = ["How many times does the robot alternate between turning and moving forward?",
          "How often does the robot switch between rotation and forward movement?",
          "Count the number of motion-mode changes in this sequence.",
          "How many transitions occur between turning and forward movement?",
          "How frequently does the robot change between translating and rotating?",
          "How many times does the robot shift from one motion type to another?"]
    n = s["switches"]
    a = (f"The robot switches between turning and forward movement {n} time{'s' if n>1 else ''}, "
         f"producing {s['n_segs']} distinct motion segment{'s' if s['n_segs']>1 else ''}.")
    return qs[v % 6], a

# 9 ── dominant motion type
@cat
def dominant_motion(s, v):
    if not (s["has_fwd"] and s["has_turns"]): return None
    qs = ["What is the dominant motion type in this sequence?",
          "Does the robot spend more steps turning or moving forward?",
          "Which motion occurs most frequently — rotation or forward movement?",
          "Is this sequence dominated by turning or by forward travel?",
          "What type of motion makes up the majority of this sequence?",
          "Between turning and moving forward, which takes up more steps?"]
    turn_steps = s["left"] + s["right"]; fwd_steps = s["fwd"]
    if fwd_steps > turn_steps:
        a = (f"Forward movement dominates: {fwd_steps} forward steps ({s['fwd_cm']}cm) "
             f"vs {turn_steps} rotation steps ({s['total_rot']}°).")
    else:
        a = (f"Turning dominates: {turn_steps} rotation steps ({s['total_rot']}°) "
             f"vs {fwd_steps} forward steps ({s['fwd_cm']}cm).")
    return qs[v % 6], a

# 10 ── has forward motion
@cat
def has_forward(s, v):
    qs = ["Does the robot move forward at all in this sequence?",
          "Is there any forward translation in this motion sequence?",
          "Does the robot translate forward, or does it only rotate?",
          "Is forward movement present in this sequence?",
          "Does this sequence include any forward steps?",
          "Does the robot cover any forward distance in this sequence?"]
    if s["has_fwd"] and s["has_turns"]:
        a = (f"The robot does both — it moves forward {s['fwd_cm']}cm ({s['fwd']} steps) "
             f"and also rotates a total of {s['total_rot']}° "
             f"({s['left_deg']}° left, {s['right_deg']}° right).")
    elif s["has_fwd"]:
        a = (f"The robot only moves forward — it covers {s['fwd_cm']}cm across "
             f"{s['fwd']} step{'s' if s['fwd']>1 else ''} with no rotation.")
    else:
        a = "The robot only rotates; there is no forward movement in this sequence."
    return qs[v % 6], a

# 11 ── final heading / orientation
@cat
def final_heading(s, v):
    if not s["has_turns"]: return None
    qs = ["What is the robot's final orientation relative to its starting heading?",
          "By how many degrees has the robot's heading changed from the start to end?",
          "What is the robot's net angular displacement after the full sequence?",
          "How does the robot's facing direction at the end compare to its initial heading?",
          "What is the net change in the robot's heading direction?",
          "After completing all turns, how far has the robot rotated from its initial direction?"]
    net = s["net_rot"]
    if net == 0:
        a = "The robot returns to its original heading — the left and right rotations cancel exactly."
    else:
        d = _d(net)
        a = (f"The robot's final heading is {abs(net)}° to the {d} of its starting orientation "
             f"(from {s['left_deg']}° left and {s['right_deg']}° right turns).")
    return qs[v % 6], a

# 12 ── sequence length
@cat
def sequence_length(s, v):
    qs = ["How many discrete motion steps does this sequence contain?",
          "What is the total number of motion commands in this sequence?",
          "How long is this motion sequence in terms of individual steps?",
          "How many individual movement tokens are encoded in this sequence?",
          "What is the length of this GRU motion sequence?",
          "How many motion tokens make up this sequence in total?"]
    n = s["seq_len"]
    a = (f"The sequence contains {n} motion step{'s' if n>1 else ''} in total, "
         f"including the initial stop token.")
    return qs[v % 6], a

# 13 ── first action after stop
@cat
def first_action(s, v):
    if s["first_code"] is None: return None
    qs = ["What is the robot's first action after starting?",
          "How does the robot begin its movement?",
          "What is the very first motion the robot performs?",
          "Describe the opening action of this motion sequence.",
          "What does the robot do immediately after the initial stop?",
          "Which movement initiates the robot's trajectory?"]
    c, n = s["first_code"], s["segs"][0][1]
    if c == 1:   a = f"The robot immediately moves forward {n*25}cm without any prior rotation."
    elif c == 2: a = f"The robot begins by turning left {n*15}° before any forward movement."
    else:        a = f"The robot begins by turning right {n*15}° before any forward movement."
    return qs[v % 6], a

# 14 ── longest consecutive forward run
@cat
def longest_forward_run(s, v):
    if not s["has_fwd"] or s["max_fwd_run"] < 2: return None
    qs = ["What is the longest uninterrupted forward movement in this sequence?",
          "How far does the robot travel in its longest continuous forward segment?",
          "What is the maximum number of consecutive forward steps the robot takes?",
          "What is the longest straight-line forward burst in this sequence?",
          "In the longest continuous forward run, how far does the robot go?",
          "What is the greatest distance the robot covers without interruption?"]
    n = s["max_fwd_run"]
    a = f"The robot's longest uninterrupted forward segment is {n*25}cm ({n} consecutive steps)."
    return qs[v % 6], a

# 15 ── longest consecutive turning run
@cat
def longest_turn_run(s, v):
    if not s["has_turns"] or s["max_turn_run"] < 2: return None
    qs = ["What is the longest consecutive turning sequence the robot performs?",
          "What is the maximum number of consecutive rotation steps in this sequence?",
          "How large is the biggest single turning maneuver the robot makes?",
          "What is the robot's longest continuous rotation segment?",
          "In the longest uninterrupted turn, how many degrees does the robot rotate?",
          "What is the greatest angle the robot rotates without interruption?"]
    n = s["max_turn_run"]
    # determine which direction
    max_l, max_r = s["max_left_run"], s["max_right_run"]
    if max_l >= max_r:
        a = f"The longest continuous rotation is {n*15}° to the left ({n} consecutive steps)."
    else:
        a = f"The longest continuous rotation is {n*15}° to the right ({n} consecutive steps)."
    return qs[v % 6], a

# 16 ── number of separate forward segments
@cat
def n_forward_segments(s, v):
    if s["n_fwd_segs"] < 2: return None
    qs = ["How many separate forward movement segments are there in this sequence?",
          "Into how many distinct bursts is the robot's forward travel divided?",
          "How many times does the robot start a new forward movement segment?",
          "How many non-contiguous forward movement periods occur in this sequence?",
          "How often does the robot resume forward movement after a turn?",
          "How many individual forward travel episodes does this sequence contain?"]
    n = s["n_fwd_segs"]
    a = (f"There are {n} separate forward movement segments, "
         f"totalling {s['fwd_cm']}cm across {s['fwd']} steps.")
    return qs[v % 6], a

# 17 ── number of separate turning segments
@cat
def n_turn_segments(s, v):
    if s["n_turn_segs"] < 2: return None
    qs = ["How many separate turning segments does the robot perform?",
          "How many times does the robot initiate a new turning maneuver?",
          "Into how many distinct rotation episodes is the robot's turning divided?",
          "How many separate rotation segments are present in this sequence?",
          "How often does the robot perform a fresh turning movement?",
          "How many individual rotation bursts occur throughout this sequence?"]
    n = s["n_turn_segs"]
    a = (f"The robot performs {n} separate turning segments, "
         f"totalling {s['total_rot']}° of rotation.")
    return qs[v % 6], a

# 18 ── forward fraction
@cat
def forward_fraction(s, v):
    if s["total_move"] == 0: return None
    if not s["has_fwd"] or not s["has_turns"]: return None  # trivial
    qs = ["What fraction of the robot's motion steps involve forward movement?",
          "What percentage of non-stop steps are forward steps?",
          "How much of the robot's motion is dedicated to moving forward?",
          "What proportion of the robot's steps are forward rather than turning?",
          "Of all active motion steps, how many are forward steps?",
          "What share of the robot's total steps are dedicated to forward movement?"]
    frac = s["fwd_frac"]
    pct = round(frac * 100)
    a = (f"{s['fwd']} out of {s['total_move']} active steps ({pct}%) are forward steps; "
         f"the remaining {100-pct}% are turning steps.")
    return qs[v % 6], a

# 19 ── does sequence end with forward or turn
@cat
def ends_with(s, v):
    if s["last_code"] is None: return None
    qs = ["Does the robot's sequence end with a forward step or a turning step?",
          "Is the final motion in this sequence a rotation or a forward movement?",
          "Does the robot finish by moving forward or by turning?",
          "What type of motion — translation or rotation — concludes the sequence?",
          "Does the robot end its movement by stepping forward or rotating?",
          "Is the last command in this sequence a turn or a forward movement?"]
    if s["ends_fwd"]:
        a = f"The sequence ends with a forward movement — the robot steps forward {s['last_n']*25}cm."
    else:
        d = "left" if s["last_code"] == 2 else "right"
        a = f"The sequence ends with a turning movement — the robot rotates {d} {s['last_n']*15}°."
    return qs[v % 6], a

# 20 ── does sequence start with forward or turn
@cat
def starts_with(s, v):
    if s["first_code"] is None: return None
    qs = ["Does the robot begin its movement with a forward step or a turning step?",
          "Is the first action in this sequence a rotation or a forward movement?",
          "Does the robot start by moving forward or by turning?",
          "What type of motion — translation or rotation — opens the sequence?",
          "Does the robot initiate movement by stepping forward or by rotating?",
          "Is the opening command in this sequence a turn or a forward step?"]
    if s["starts_fwd"]:
        a = "The sequence opens with forward movement — the robot begins translating immediately."
    else:
        d = "left" if s["first_code"] == 2 else "right"
        a = f"The sequence opens with a {d} turn — the robot rotates before moving forward."
    return qs[v % 6], a

# 21 ── left vs right breakdown
@cat
def left_right_breakdown(s, v):
    if not (s["has_left"] and s["has_right"]): return None
    qs = ["How are left turns and right turns distributed in this sequence?",
          "What is the breakdown of left versus right rotations?",
          "How many degrees does the robot turn left compared to right?",
          "Compare the robot's leftward and rightward rotation in this sequence.",
          "How does the amount of left turning compare to right turning?",
          "What is the split between left-turn steps and right-turn steps?"]
    a = (f"The robot turns left {s['left_deg']}° ({s['left']} steps) "
         f"and right {s['right_deg']}° ({s['right']} steps).")
    return qs[v % 6], a

# 22 ── distance traveled after the final turn
@cat
def forward_after_last_turn(s, v):
    if not s["has_turns"] or s["fwd_after_last_turn"] == 0: return None
    if not s["has_fwd"]: return None
    qs = ["How far does the robot travel forward after its last turn?",
          "What distance does the robot cover in its final straight segment?",
          "How many centimeters does the robot move after completing its last rotation?",
          "What is the length of the robot's final forward run after the last turn?",
          "After the robot makes its final turn, how far does it continue forward?",
          "What is the distance the robot travels in a straight line at the end of the sequence?"]
    cm = s["fwd_after_last_turn"]
    a = f"After its final turn, the robot moves forward {cm}cm in a straight line."
    return qs[v % 6], a

# 23 ── total active motion steps (excluding initial stop)
@cat
def total_active_steps(s, v):
    if s["total_move"] == 0: return None
    qs = ["How many active motion steps does the sequence contain, excluding the initial stop?",
          "What is the total number of non-stop motion commands in this sequence?",
          "Excluding the starting stop token, how many motion steps are there?",
          "How many steps involve actual movement rather than stopping?",
          "What is the count of motion-executing steps in this sequence?",
          "How many steps contribute to the robot's movement, not counting the initial stop?"]
    n = s["total_move"]
    a = (f"There are {n} active motion step{'s' if n>1 else ''} — "
         f"{s['fwd']} forward and {s['left']+s['right']} rotation step{'s' if s['left']+s['right']>1 else ''}.")
    return qs[v % 6], a

# 24 ── motion complexity / structural description
@cat
def motion_complexity(s, v):
    if s["n_segs"] < 1: return None
    qs = ["How would you characterize the overall structure of this motion sequence?",
          "Is this a simple or complex motion sequence, and why?",
          "Describe the general movement pattern of this sequence.",
          "What is the high-level structure of the robot's motion?",
          "How complex is the robot's path in this sequence?",
          "Provide a structural overview of the robot's motion."]
    segs, fwd, turns = s["n_segs"], s["fwd"], s["left"]+s["right"]
    if segs == 1:
        c = s["segs"][0][0]
        if c == 1: label = "a single straight forward movement"
        elif c == 2: label = "a single left-turn maneuver with no forward movement"
        else: label = "a single right-turn maneuver with no forward movement"
        a = f"This is a simple sequence consisting of {label}."
    elif segs <= 3 and s["switches"] <= 1:
        a = (f"This is a straightforward sequence of {segs} segments: the robot "
             f"{'rotates then moves forward' if not s['starts_fwd'] else 'moves forward then rotates'}.")
    else:
        a = (f"This is a moderately complex sequence of {segs} segments, "
             f"with {s['switches']} motion-type switch{'es' if s['switches']>1 else ''}, "
             f"{fwd} forward step{'s' if fwd>1 else ''} ({s['fwd_cm']}cm), "
             f"and {turns} rotation step{'s' if turns>1 else ''} ({s['total_rot']}°).")
    return qs[v % 6], a

# 25 ── left-only or right-only turning
@cat
def single_turn_direction(s, v):
    # interesting only when the robot turns in exactly one direction
    if not s["has_turns"]: return None
    if s["has_left"] and s["has_right"]: return None  # mixed — covered elsewhere
    qs = ["Does the robot turn exclusively in one direction throughout the sequence?",
          "Is all of the robot's rotation in the same direction?",
          "Does the robot ever turn in both directions, or always the same way?",
          "In which single direction does the robot rotate in this sequence?",
          "Is the robot's turning direction consistent throughout the sequence?",
          "Does the robot make turns in only one direction from start to finish?"]
    d = "left" if s["has_left"] else "right"
    deg = s["left_deg"] if s["has_left"] else s["right_deg"]
    a = (f"Yes, the robot turns exclusively to the {d} throughout the sequence, "
         f"rotating a total of {deg}°.")
    return qs[v % 6], a


# ──────────────────────────────────────────────────────────────────────────────
# Record-level Q&A selector
# ──────────────────────────────────────────────────────────────────────────────

def _hash(record_id):
    return int(hashlib.md5(record_id.encode()).hexdigest(), 16)

def generate_qa(record_id, gru):
    # Empty GRU — no motion data available
    if not gru:
        return [
            {"from": "human",
             "value": "<image>\n<gru>\nWhat does the GRU motion sequence tell us about the robot's movement?"},
            {"from": "gpt",
             "value": "The GRU sequence is empty; no motion data is available for this entry."}
        ]

    s = analyze(gru)

    # Single stop token [0] — robot is stationary
    if s is None:
        return [
            {"from": "human",
             "value": "<image>\n<gru>\nWhat does this motion sequence indicate about the robot's state?"},
            {"from": "gpt",
             "value": "The sequence contains only a stop command; the robot remains stationary."}
        ]

    h = _hash(record_id)
    variant = (h >> 8) % 6

    # Collect all valid categories in their natural order, then rotate start
    valid = [(i, fn) for i, fn in enumerate(CATS) if fn(s, variant) is not None]
    if not valid:
        return [
            {"from": "human", "value": "<image>\n<gru>\nDescribe the robot's movement."},
            {"from": "gpt",   "value": f"The sequence has {s['seq_len']} steps."}
        ]

    # Uniform selection: pick category by hash, phrasing variant separately
    chosen_fn = valid[h % len(valid)][1]
    q, a = chosen_fn(s, variant)

    return [
        {"from": "human", "value": f"<image>\n<gru>\n{q}"},
        {"from": "gpt",   "value": a},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

INPUT_PATH  = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Dataset/test/r2r_alignment_dataset.json"
OUTPUT_PATH = "/home/djonna1/scratchtinoosh/iros_dataset/Qwen-Dataset/test/r2r_alignment_dataset_qa.json"

print(f"Processing {INPUT_PATH} → {OUTPUT_PATH} ...")

written = 0
with open(INPUT_PATH, "rb") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    fout.write("[\n")
    first = True
    for item in ijson.items(fin, "item"):
        item["conversations"] = generate_qa(item["id"], item["gru"])
        if not first:
            fout.write(",\n")
        fout.write(json.dumps(item, ensure_ascii=False))
        first = False
        written += 1
        if written % 20000 == 0:
            print(f"  {written:,} records ...")
    fout.write("\n]")

print(f"\nDone — {written:,} records saved to {OUTPUT_PATH}")
