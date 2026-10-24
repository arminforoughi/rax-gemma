"""Gemma as the robot's judgement, running on this machine.

    from rax.gemma import Gemma, Place, run_request
    run_request(arm, Gemma("gemma4:e4b"), "sort the tubes by cap colour", places)

Gemma looks and plans (:mod:`rax.gemma.brain`); RAX picks and places; Gemma checks
(:mod:`rax.gemma.task`). The model runs on this machine: a local Ollama on a laptop
(:mod:`rax.gemma.client`), or LiteRT-LM in this process on a Raspberry Pi
(:mod:`rax.gemma.litert`). Nothing goes to the cloud.
"""

from .brain import Item, Plan, Step, Verdict, check, look, plan
from .client import Gemma, GemmaError
from .litert import LiteRtGemma, open_gemma
from .task import Outcome, Place, Report, execute, run_request, see, target_for

__all__ = ["Gemma", "GemmaError", "LiteRtGemma", "open_gemma", "Item", "Plan", "Step",
           "Verdict", "look", "plan", "check", "Place", "Outcome", "Report", "run_request",
           "see", "execute", "target_for"]
