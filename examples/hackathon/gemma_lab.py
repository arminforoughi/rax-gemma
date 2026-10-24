"""Tell the arm what to do. Gemma, on this machine, looks, plans and checks; RAX picks.

On a laptop (Ollama, Gemma 4 E4B):

    ollama pull gemma4:e4b
    python examples/hackathon/gemma_lab.py --sim                 # no robot
    python examples/hackathon/gemma_lab.py --port /dev/tty.usbmodem...   # the SO-101
    python examples/hackathon/gemma_lab.py --sim --say "put the red tube in the left rack"

On a Raspberry Pi 5 (LiteRT-LM, Gemma 4 E2B, everything on the one board; see
examples/hackathon/PI.md):

    python examples/hackathon/gemma_lab.py --pi --port /dev/ttyACM0

Type a request at the prompt ("sort the tubes: red left, blue right", "what do you
see?"). Other commands:

    look        what Gemma sees, nothing moves
    plan <req>  the plan for a request, nothing moves
    places      the place names and where they are
    q           quit

Places come from --places (JSON: {"left rack": [x, y, release_z], ...}, metres in the
arm's base frame) or the defaults below. --speak reads the answers out with macOS
``say`` or, on the Pi, ``espeak-ng``: both on-device.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

from rax.gemma import GemmaError, Place, look, open_gemma, plan, run_request

#: Two drop spots on either side of the SO-101, in reach at a high release.
DEFAULT_PLACES = {"left rack": (0.20, 0.15, 0.10), "right rack": (0.20, -0.15, 0.10)}


#: What the simulator draws, so Gemma does not read its tubes as bars and balls.
SIM_HINT = ("This is a simple drawing of test tubes lying on a table, seen from above. "
            "Each light grey bar is a tube and the coloured circle at its end is its "
            "cap; call each one by its cap colour, e.g. 'red tube'.")


def load_places(path: str | None, sim: bool) -> list[Place]:
    spec = DEFAULT_PLACES
    if path:
        with open(path) as f:
            spec = json.load(f)
    # in the simulator a tube is stood up and dropped, as examples/pick_demo.py does
    extra = {"pitch": 0.0, "roll": 90.0} if sim else {}
    return [Place(name, float(v[0]), float(v[1]), float(v[2]), **extra)
            for name, v in spec.items()]


def make_arm(args):
    if args.sim:
        from rax.pick.sim import SimArm, Tube
        return SimArm([Tube(0.24, 0.10, 30, "green"), Tube(0.27, 0.02, 120, "blue"),
                       Tube(0.21, 0.05, 250, "red")])
    import atexit

    from rax.robots.so101 import So101
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    handeye = args.handeye or next(p for p in (os.path.join(here, "handeye_so101.json"),
                                               os.path.join(here, "handeye_so101.example.json"))
                                   if os.path.exists(p))
    arm = So101(args.port, handeye_file=handeye)
    print(f"connecting the SO-101 on {args.port} (hand-eye: {os.path.basename(handeye)}) ...")
    arm.connect()
    atexit.register(arm.disconnect)
    return arm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", action="store_true", help="the simulated SO-101")
    ap.add_argument("--port", help="the SO-101's serial port")
    ap.add_argument("--handeye", help="hand-eye calibration JSON for the real arm")
    ap.add_argument("--pi", action="store_true",
                    help="Raspberry Pi preset: --backend litert, the colour detector")
    ap.add_argument("--backend", choices=("ollama", "litert"),
                    help="ollama (a laptop, default) or litert (LiteRT-LM in this process)")
    ap.add_argument("--model", default=os.environ.get("RAX_GEMMA"),
                    help="ollama tag (gemma4:e4b) or .litertlm path "
                         "(~/models/gemma-4-E2B-it.litertlm)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--litert-backend", default="cpu", choices=("cpu", "gpu", "npu"))
    ap.add_argument("--places", help="JSON of named places")
    ap.add_argument("--detector", choices=("prompt", "colour"),
                    help="how the arm finds what Gemma named (default: colour in --sim)")
    ap.add_argument("--hint", help="what Gemma should expect on the table")
    ap.add_argument("--no-verify", action="store_true", help="skip Gemma's check")
    ap.add_argument("--speak", action="store_true", help="say answers out loud")
    ap.add_argument("--say", dest="request", help="do this one request and exit")
    args = ap.parse_args()
    if not args.sim and not args.port:
        ap.error("--sim or --port")
    backend = args.backend or ("litert" if args.pi else "ollama")
    # YOLO-World is slow on the Pi's CPU; the cap colours need no model at all
    detector = args.detector or ("colour" if args.sim or args.pi else "prompt")
    hint = args.hint if args.hint is not None else (SIM_HINT if args.sim else "")

    talk = (shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")) \
        if args.speak else None

    def say(msg):
        print(msg, flush=True)
        if talk and not msg.startswith("  "):          # the sentences, not the detail
            subprocess.run([talk, msg])

    try:
        gemma = (open_gemma("litert", args.model, backend=args.litert_backend)
                 if backend == "litert" else open_gemma("ollama", args.model, host=args.host))
    except GemmaError as e:
        sys.exit(str(e))
    if not gemma.ready():
        sys.exit(gemma.why_not_ready())
    print(f"loading {gemma.model} ({backend})...", flush=True)
    print(f"  ready in {gemma.warm():.0f}s")
    places = load_places(args.places, args.sim)
    arm = make_arm(args)
    print(f"{gemma.model} ({backend}); detector: {detector}; "
          f"places: {', '.join(p.name for p in places)}")

    def do(req):
        try:
            run_request(arm, gemma, req, places, detector=detector,
                        verify=not args.no_verify, say=say, hint=hint)
        except (GemmaError, RuntimeError) as e:
            say(f"stopped: {e}")

    if args.request:
        do(args.request)
        return
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            break
        if line == "places":
            for p in places:
                print(f"  {p.name}: ({p.x*100:.0f}, {p.y*100:.0f}) cm, release at "
                      f"{p.release_z*100:.0f} cm")
        elif line == "look":
            items = look(gemma, arm.frame(), hint)
            say(", ".join(f"{i.name} ({i.where})" for i in items) or "nothing")
        elif line.startswith("plan "):
            items = look(gemma, arm.frame(), hint)
            p = plan(gemma, line[5:], items, [pl.name for pl in places])
            say(p.say)
            for k, s in enumerate(p.steps, 1):
                print(f"  {k}. {s.object} -> {s.to}")
        else:
            do(line)


if __name__ == "__main__":
    main()
