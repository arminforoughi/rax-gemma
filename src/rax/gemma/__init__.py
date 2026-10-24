"""Gemma as the robot's judgement, running on this machine.

    from rax.gemma import Gemma, Place, run_request
    run_request(arm, Gemma("gemma4:e4b"), "sort the tubes by cap colour", places)

Gemma looks and plans (:mod:`rax.gemma.brain`); RAX picks and places; Gemma checks
(:mod:`rax.gemma.task`). The model is served by a local Ollama (:mod:`rax.gemma.client`)
and nothing goes to the cloud.
"""

from .brain import Item, Plan, Step, Verdict, check, look, plan
from .client import Gemma, GemmaError
from .task import Outcome, Place, Report, run_request, target_for

__all__ = ["Gemma", "GemmaError", "Item", "Plan", "Step", "Verdict", "look", "plan",
           "check", "Place", "Outcome", "Report", "run_request", "target_for"]
