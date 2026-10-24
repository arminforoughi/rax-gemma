"""What Gemma decides: what is on the table, what to do about a request, did it work.

Three questions, each one model call with a JSON schema:

* :func:`look`: list the objects in a picture, each named so a detector can find it
  ("red cube", "blue-capped tube"), not described ("a small reddish thing").
* :func:`plan`: turn a spoken request and that list into pick-and-place steps, each
  onto one of the named places. Steps naming a place that does not exist are dropped
  and said, not guessed at.
* :func:`check`: a yes/no question about a picture ("is the tube in the rack?").

Gemma never steers the arm: the pick is RAX's visual servo, which runs at camera rate.
Gemma is asked once per task and once per check, so a few seconds an answer is fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SYSTEM = ("You are the eyes and judgement of a small robot arm on a lab bench. "
          "You see through its wrist camera. Be literal and brief. Only report what "
          "you can actually see.")


@dataclass
class Item:
    name: str                     # detector-friendly: "red cube", "blue-capped tube"
    colour: str = ""
    where: str = ""               # "left", "front right", ...


@dataclass
class Step:
    object: str                   # what to pick, as a detector prompt
    to: str                       # one of the place names


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    say: str = ""                 # one sentence back to the person
    dropped: list[str] = field(default_factory=list)   # steps that could not be kept


LOOK_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "colour": {"type": "string"},
                       "where": {"type": "string"}},
        "required": ["name", "colour", "where"]}}},
    "required": ["objects"],
}


def look(gemma, bgr, hint: str = "") -> list[Item]:
    """The objects Gemma sees in ``bgr`` that an arm could pick up.

    ``hint`` says what to expect ("sample tubes with coloured caps"), which settles
    what an ambiguous shape is.
    """
    out = gemma.ask(
        (f"{hint.strip()}\n\n" if hint else "") +
        "List every small object on the table that a robot gripper could pick up. "
        "For each, give a short name a detector could search for, with its colour "
        "first (e.g. 'red cube', 'green-capped tube', 'yellow ball'), its main colour, "
        "and roughly where it is in the picture (left, centre, right, front, back). "
        "Only list objects lying loose on the table or mat. Do not list anything "
        "already standing in a rack or holder, and do not list the table, the mat, "
        "racks, or the robot itself. List each object once.",
        images=[bgr], schema=LOOK_SCHEMA, system=SYSTEM)
    items = []
    for o in out.get("objects", []):
        name = _clean(o.get("name", ""))
        if name:
            items.append(Item(name, _clean(o.get("colour", "")), _clean(o.get("where", ""))))
    return items


def plan_schema(places: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "say": {"type": "string", "minLength": 3},
            "steps": {"type": "array", "items": {
                "type": "object",
                "properties": {"object": {"type": "string"},
                               "to": {"type": "string", "enum": list(places)}},
                "required": ["object", "to"]}},
        },
        "required": ["say", "steps"],
    }


def plan(gemma, request: str, items: list[Item], places: list[str], bgr=None,
         history: str = "") -> Plan:
    """Steps that carry out ``request`` with the objects in ``items``.

    ``history`` is the conversation so far ("it", "the other one"), as plain lines.
    ``places`` are where things may go. Every step's object is a name from ``items``
    (or close to one) so the detector can find it; every step's place is one of
    ``places``. ``bgr``, if given, is shown too, for requests about what things look
    like rather than what they are called.
    """
    seen = "\n".join(f"- {i.name} ({i.where})" for i in items) or "- nothing"
    out = gemma.ask(
        (f"Earlier in this conversation:\n{history}\n\n" if history else "") +
        f"The person says: \"{request}\"\n\n"
        f"Objects on the table:\n{seen}\n\n"
        f"Places things can go: {', '.join(places)}.\n\n"
        "Write the pick-and-place steps that do what they asked, in order, one object "
        "per step. Use the object names exactly as listed. If the same kind of object "
        "appears more than once, write one step per object. Use no steps if nothing "
        "needs moving or the request is a question. In 'say', first, answer them in "
        "one short spoken sentence: what you will do, or the answer to their "
        "question, or why you cannot.",
        images=[bgr] if bgr is not None else (), schema=plan_schema(places), system=SYSTEM)
    p = Plan(say=_clean(out.get("say", "")))
    names = [i.name for i in items]
    for s in out.get("steps", []):
        obj, to = _clean(s.get("object", "")), _clean(s.get("to", ""))
        if to not in places:
            p.dropped.append(f"{obj} -> {to} (no such place)")
        elif not obj:
            p.dropped.append(f"? -> {to} (no object)")
        else:
            p.steps.append(Step(_closest(obj, names), to))
    return p


CHECK_SCHEMA = {
    "type": "object",
    "properties": {"yes": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["yes", "why"],
}


@dataclass
class Verdict:
    yes: bool
    why: str


def check(gemma, bgr, question: str) -> Verdict:
    """Gemma's yes or no to ``question`` about ``bgr``, with a one-line reason."""
    out = gemma.ask(f"{question} Answer yes or no, and say why in a few words.",
                    images=[bgr], schema=CHECK_SCHEMA, system=SYSTEM)
    return Verdict(bool(out.get("yes")), _clean(out.get("why", "")))


def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").strip()


def _closest(name: str, names: list[str]) -> str:
    """``name`` as written in ``names`` if it is one, ignoring case and articles."""
    def key(s):
        return re.sub(r"\b(the|a|an)\b|\s+", " ", s.lower()).strip()
    for n in names:
        if key(n) == key(name):
            return n
    return name
