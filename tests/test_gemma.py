"""rax.gemma: the plan is kept honest, and a request runs end to end on the simulator.

Gemma itself is replaced by a scripted stand-in that answers by schema, so these pin
what RAX does with the answers, not what the model says.
"""

import pytest

from rax.gemma import GemmaError, Place, plan, run_request, target_for
from rax.gemma.brain import CHECK_SCHEMA, LOOK_SCHEMA, Item
from rax.gemma.client import Gemma, is_local
from rax.pick import ColourTarget
from rax.pick.sim import SimArm, Tube


class FakeGemma:
    """Answers each kind of question with a canned reply; records the prompts."""

    def __init__(self, objects=(), steps=(), say="ok", checks=(True,)):
        self.objects, self.steps, self.say_ = list(objects), list(steps), say
        self.checks = list(checks)
        self.prompts = []
        self.last_s = 0.0

    def ask(self, prompt, images=(), schema=None, system=None):
        self.prompts.append(prompt)
        if schema is LOOK_SCHEMA:
            return {"objects": [{"name": n, "colour": n.split()[0], "where": "centre"}
                                for n in self.objects]}
        if schema is CHECK_SCHEMA:
            yes = self.checks.pop(0) if len(self.checks) > 1 else self.checks[0]
            return {"yes": yes, "why": "seen" if yes else "not there"}
        return {"steps": [{"object": o, "to": t} for o, t in self.steps], "say": self.say_}


PLACES = ["left rack", "right rack"]


def test_plan_drops_a_place_that_does_not_exist():
    g = FakeGemma(steps=[("red tube", "left rack"), ("blue tube", "the bin")])
    p = plan(g, "tidy up", [Item("red tube"), Item("blue tube")], PLACES)
    assert [(s.object, s.to) for s in p.steps] == [("red tube", "left rack")]
    assert p.dropped and "the bin" in p.dropped[0]


def test_plan_uses_the_seen_name_for_the_detector():
    g = FakeGemma(steps=[("The Red Tube", "left rack")])
    p = plan(g, "red left", [Item("red tube")], PLACES)
    assert p.steps[0].object == "red tube"


def test_target_for_colour_needs_a_colour():
    t = target_for("green tube", "colour")
    assert isinstance(t, ColourTarget) and t.colours == ("green",)
    with pytest.raises(RuntimeError):
        target_for("tube", "colour")


def test_cloud_and_remote_hosts_are_refused():
    assert is_local("http://127.0.0.1:11434")
    assert not is_local("http://8.8.8.8:11434")
    with pytest.raises(GemmaError):
        Gemma("gemma4:31b-cloud")
    with pytest.raises(GemmaError):
        Gemma("gemma4:e4b", host="http://8.8.8.8:11434")


def _sim():
    return SimArm([Tube(0.24, 0.10, 30, "green"), Tube(0.27, 0.02, 120, "blue")])


SIM_PLACES = [Place("left rack", 0.20, 0.15, 0.10, pitch=0.0, roll=90.0),
              Place("right rack", 0.20, -0.15, 0.10, pitch=0.0, roll=90.0)]


def test_a_request_runs_on_the_simulator():
    arm = _sim()
    g = FakeGemma(objects=["green tube", "blue tube"],
                  steps=[("green tube", "left rack"), ("blue tube", "right rack")])
    rep = run_request(arm, g, "green left, blue right", SIM_PLACES, detector="colour",
                      say=lambda m: None)
    assert [o.ok for o in rep.outcomes] == [True, True], [o.detail for o in rep.outcomes]
    # the tubes left the table: none is still where it started
    assert all(abs(t.x - 0.24) > 0.02 or abs(t.y - 0.10) > 0.02 for t in arm.tubes
               if t.colour == "green")


def test_a_text_only_gemma_plans_from_what_the_detector_sees():
    """Gemma 3 1B cannot look: the detector lists the table, Gemma only plans, no checks."""
    class TextOnly(FakeGemma):
        vision = False

        def ask(self, prompt, images=(), schema=None, system=None):
            assert not images and schema not in (LOOK_SCHEMA, CHECK_SCHEMA)
            return super().ask(prompt, images, schema, system)

    g = TextOnly(steps=[("green", "left rack")])
    rep = run_request(_sim(), g, "green left", SIM_PLACES, detector="colour",
                      target=ColourTarget(colours=("green", "blue")), say=lambda m: None)
    assert {i.name for i in rep.seen} == {"green", "blue"}
    assert "- green (" in g.prompts[0]            # the plan was asked with the detector's list
    assert [o.ok for o in rep.outcomes] == [True], [o.detail for o in rep.outcomes]
    assert "placed (jaws" in rep.outcomes[0].detail   # placed, and not checked


def test_a_question_or_hello_never_moves_the_arm():
    """Gemma 4 E2B text-only planned moves for both of these on the Pi."""
    from rax.gemma.brain import asks_to_move
    for q in ("what do you see?", "hello, how are you", "is the red tube in the rack",
              "can you see the blue tube?", "thanks"):
        assert not asks_to_move(q), q
        p = plan(FakeGemma(steps=[("red tube", "left rack")], say="hi"), q,
                 [Item("red tube")], PLACES)
        assert not p.steps and p.say == "hi" and p.dropped, q
    for r in ("put the red tube in the left rack", "red left", "can you move it right?",
              "sort the tubes"):
        assert asks_to_move(r), r


def test_text_only_models_are_told_by_name():
    from rax.gemma.client import is_vision_model
    for name in ("gemma3-1b-it-int4.litertlm", "gemma3-270m-it-q8.litertlm", "gemma3:1b"):
        assert not is_vision_model(name), name
    for name in ("gemma-4-E2B-it.litertlm", "gemma4:e4b", "gemma3:12b"):
        assert is_vision_model(name), name


def test_a_failed_check_retries_then_reports():
    g = FakeGemma(objects=["green tube"], steps=[("green tube", "left rack")],
                  checks=(False,))
    rep = run_request(_sim(), g, "green left", SIM_PLACES, detector="colour", retries=1,
                      say=lambda m: None)
    assert not rep.ok and "Gemma does not see it" in rep.outcomes[0].detail


def test_an_object_not_on_the_table_is_not_retried():
    g = FakeGemma(objects=["red tube"], steps=[("red tube", "left rack")])
    rep = run_request(_sim(), g, "red left", SIM_PLACES, detector="colour",
                      say=lambda m: None)
    assert not rep.ok and "not on the table" in rep.outcomes[0].detail
    assert sum("Is there" in p for p in g.prompts) == 0


# ---- the LiteRT-LM backend (the Pi): no model needed for these ----

def test_first_json_takes_the_object_out_of_a_fenced_answer():
    from rax.gemma.litert import first_json
    assert first_json('Sure!\n```json\n{"yes": true, "why": "a {brace} in a string"}\n```') == \
        {"yes": True, "why": "a {brace} in a string"}
    with pytest.raises(GemmaError):
        first_json("no json here")
    with pytest.raises(GemmaError):
        first_json('{"yes": true, "why": "cut')


def test_open_gemma_picks_the_backend():
    from rax.gemma import LiteRtGemma, open_gemma
    g = open_gemma("litert", "/nonexistent/model.litertlm")
    assert isinstance(g, LiteRtGemma) and not g.ready()
    assert "model.litertlm" in g.why_not_ready() or "pip install" in g.why_not_ready()
    assert isinstance(open_gemma("ollama", "gemma4:e4b"), Gemma)
