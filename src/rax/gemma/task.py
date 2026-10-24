"""Do what the person asked: Gemma looks and plans, RAX picks and places, Gemma checks.

    from rax.gemma import Gemma, Place, run_request

    places = [Place("left tray", 0.20, -0.15, 0.06), Place("right tray", 0.20, 0.15, 0.06)]
    run_request(arm, Gemma(), "put the red things in the left tray", places)

For each step of Gemma's plan the arm maps the table for that object (one sweep of the
wrist camera with the step's name as the detector prompt), picks the best-seen one,
carries it to the place and lets go. Then, from above the place, Gemma is asked
whether it landed. A failed pick is tried again; a "no" from Gemma is reported, not
retried, since a rescan would take the object back out of its place. Every failure
is said, not hidden.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from rax.pick import TOP, ColourTarget, Grasp, PromptTarget, StickyTarget, pick, place, scan
from rax.pick.arm import steepest
from rax.pick.targets import colour_names

from .brain import Item, Plan, Step, check, look, plan


@dataclass
class Place:
    """Somewhere things can go, named the way a person would say it."""

    name: str
    x: float
    y: float
    release_z: float = 0.06           # fingertip height to let go at, metres
    pitch: float | None = None        # hand angle there; None: as it was picked
    roll: float | None = None         # wrist roll there; None: as it was picked


@dataclass
class Outcome:
    step: Step
    ok: bool
    detail: str
    seconds: float = 0.0


@dataclass
class Report:
    request: str
    seen: list[Item] = field(default_factory=list)
    plan: Plan | None = None
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.plan) and all(o.ok for o in self.outcomes)


def target_for(name: str, detector: str = "prompt", grasp_z: float = 0.02):
    """A detector for the object Gemma named.

    ``"prompt"``: YOLO-World with the name as its prompt (and the colour blob when the
    name has a colour and YOLO misses it). ``"colour"``: only the colour in the name,
    no model at all; it is what the simulator draws and the fallback when the room's
    light confuses YOLO.
    """
    if detector == "colour":
        cols = tuple(colour_names(name))
        if not cols:
            raise RuntimeError(f"'{name}' names no colour for the colour detector")
        return ColourTarget(name=name, colours=cols, grasp_z=grasp_z)
    return StickyTarget(PromptTarget(name=name, prompt=name, grasp_z=grasp_z))


def run_request(arm, gemma, request: str, places: list[Place], detector: str = "prompt",
                grasp: Grasp = TOP, verify: bool = True, retries: int = 1, scan_kw=None,
                say=print, plan_only: bool = False, hint: str = "") -> Report:
    """Look, plan, then pick and place each step. Returns what happened."""
    rep = Report(request)
    by_name = {p.name: p for p in places}
    arm.move(np.array(arm.home, float), speed=1.4, settle=0.3)
    bgr = _frame(arm)

    arm.phase("LOOK", "asking Gemma what is on the table")
    rep.seen = look(gemma, bgr, hint)
    if detector == "colour":                       # what the cap finder sees counts too
        names = {i.name.lower() for i in rep.seen}
        for d in ColourTarget(colours=("green", "blue", "red")).detect(bgr):
            name = f"{d.label} tube"
            if name not in names:
                names.add(name)
                rep.seen.append(Item(name, d.label, "seen by the colour detector"))
    say(f"I see: {', '.join(i.name for i in rep.seen) or 'nothing'}  ({gemma.last_s:.1f}s)")

    arm.phase("PLAN", f"asking Gemma how to: {request}")
    rep.plan = plan(gemma, request, rep.seen, list(by_name))
    say(rep.plan.say or "(no answer)")
    for d in rep.plan.dropped:
        say(f"  skipping {d}")
    for k, s in enumerate(rep.plan.steps, 1):
        say(f"  {k}. {s.object} -> {s.to}")
    if plan_only:
        return rep

    for s in rep.plan.steps:
        arm.checkpoint()
        t0 = time.time()
        for attempt in range(retries + 1):
            ok, detail = _do_step(arm, gemma, s, by_name[s.to], detector, grasp, verify,
                                  scan_kw or {}, say, hint)
            # only a failed pick is tried again: after a place, a rescan would find the
            # object in its new spot and take it back out
            if ok or not detail.startswith("pick"):
                break
            if attempt < retries:
                say(f"  {s.object}: {detail}; trying again")
        rep.outcomes.append(Outcome(s, ok, detail, time.time() - t0))
        say(f"  {'done' if ok else 'FAILED'}: {s.object} -> {s.to}: {detail}")
    arm.move(np.array(arm.home, float), speed=1.4, settle=0.2)
    n_ok = sum(o.ok for o in rep.outcomes)
    if rep.outcomes:
        say(f"{n_ok} of {len(rep.outcomes)} done")
    return rep


def _do_step(arm, gemma, s: Step, dest: Place, detector, grasp, verify, scan_kw, say,
             hint=""):
    """One object, table to place. (ok, what happened)."""
    try:
        target = target_for(s.object, detector)
    except RuntimeError as e:
        return False, str(e)
    found = scan(arm, target, **scan_kw)
    if not found:
        return False, "not on the table (the detector found none)"
    best = max(found, key=lambda f: f.n)
    try:
        res = pick(arm, target, near_xy=(best.x, best.y), label=best.label, grasp=grasp)
    except RuntimeError as e:
        _let_go(arm)
        return False, f"pick: {e}"
    roll = float(arm.joints()[arm.roll]) if dest.roll is None else dest.roll
    pitch = res.pitch if dest.pitch is None else dest.pitch
    try:
        place(arm, (dest.x, dest.y), release_z=dest.release_z, pitch=pitch, roll=roll)
    except RuntimeError as e:
        _let_go(arm)
        return False, f"place: {e}"
    if not verify:
        return True, f"placed (jaws held at {res.grip_pct:.0f})"
    arm.phase("CHECK", f"asking Gemma if the {s.object} is in the {dest.name}")
    _look_down_at(arm, dest)
    v = check(gemma, _frame(arm),
              (f"{hint.strip()} " if hint else "") +
              f"The robot just put a {s.object} down in the {dest.name}, and this is "
              f"its camera looking at that spot. Can you see the {s.object}?")
    say(f"  check: {'yes' if v.yes else 'no'}, {v.why}  ({gemma.last_s:.1f}s)")
    return v.yes, ("placed, Gemma sees it" if v.yes else f"Gemma does not see it: {v.why}")


def _look_down_at(arm, dest: Place):
    """Point the wrist camera down at the place, so it sees what landed there.

    A place made with a level hand leaves the camera looking at the far wall. Tries a
    few hand positions above and short of the place, at the steepest angle each
    reaches, and keeps the one that puts the place nearest the middle of the image.
    """
    roll = float(arm.joints()[arm.roll])
    w, h = arm.frame().shape[1::-1] if arm.frame() is not None else (640, 480)
    spot = (dest.x, dest.y, dest.release_z)
    r = math.hypot(dest.x, dest.y)
    best = None
    for up in (0.08, 0.12, 0.16):
        for back in (0.0, 0.04, 0.08):
            k = max(0.0, (r - back) / r)
            q, _ = steepest(arm, (dest.x * k, dest.y * k, dest.release_z + up),
                            TOP.pitch, 30.0, roll, tol=0.01)
            uv = arm.project(spot, q) if q is not None else None
            if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                continue
            off = math.hypot(uv[0] - w / 2, uv[1] - h / 2)
            if best is None or off < best[0]:
                best = (off, q)
    if best is None:
        arm.log("        cannot point the camera at the place; checking the current view")
        return
    arm.move(best[1], speed=1.2, settle=0.3)


def _frame(arm, tries: int = 10):
    for _ in range(tries):
        bgr = arm.frame()
        if bgr is not None:
            return bgr
        time.sleep(0.1)
    raise RuntimeError("no camera frame")


def _let_go(arm):
    try:
        arm.release()
    except Exception:
        pass
