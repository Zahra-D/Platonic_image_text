"""Lexically matched binding-swap caption sets for held-out CLEVR worlds.

Captions follow the human-style caption generator's phrase styles, openers,
and synonym pools (``generate_human_captions.py``), but every random choice is
fixed by a *phrasing plan* instead of drawn while rendering:

* one synonym per attribute value (every blue object says "blue", every
  sphere says the same noun) for the whole caption, with one material word
  that is valid in both adjective and noun slots;
* one object-phrase style per inventory *position*;
* one object-phrase style per relation *mention* (subject or anchor);
* one relation phrase, comparative, and opener per relation, independent
  of where that relation's sentence is placed;
* one inventory opener, list joiner, and digit/word count choice.

Because words attach to values and styles attach to positions, two
renderings with the same plan contain the same multiset of words whenever the
same attribute values and relation types occur, however they are bound to
objects.  That is exactly what a binding swap preserves, so bag-of-words and
bag-of-embeddings baselines cannot tell the query and its swapped negative
apart.  The token multisets are verified for every item, not assumed.  Two
vowel-initial synonyms are excluded so that "a"/"an" never changes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from collections import Counter

COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
SHAPES = ("cube", "sphere", "cylinder")
MATERIALS = ("metal", "rubber")
SIZES = ("small", "large")
ATTRIBUTES = ("color", "shape", "material", "size")
SWAP_TYPES = ATTRIBUTES + ("relation",)
N_OBJECT_STYLES = 6


def load_generator(path: str):
    spec = importlib.util.spec_from_file_location("human_caption_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def consonant_initial(pool: list[str]) -> list[str]:
    """Drop vowel-initial synonyms ("oversized", "orb").

    Object phrases start with an article chosen from their first word, so a
    vowel-initial synonym would turn "a" into "an" whenever a swap or a
    reordering moved it to the front of a phrase, and the two captions would
    no longer have identical tokens.  With these two words excluded every
    inventory article is "a" and no item has to be discarded.
    """
    kept = [word for word in pool if word[0].lower() not in "aeiou"]
    return kept or pool


LEXICON_KEYS = ("size", "color", "shape", "material")


def make_plan(generator, seed: int, n_objects: int, n_relations: int,
              lexicon: dict | None = None) -> dict:
    """A phrasing plan: which synonym stands for each attribute value, and how
    the sentences are built.

    ``lexicon`` copies the attribute vocabulary of another plan, so the two
    plans differ only in *structure* -- phrase styles, relation wording,
    openers, ordering.  The synonym draws still happen, so every other field
    is bit-identical to the same call without ``lexicon``, and the two
    conditions differ in nothing else.
    """
    rng = random.Random(seed)
    pick = lambda pool: rng.choice(consonant_initial(pool))
    plan = {
        "size": {v: pick(generator.SIZE_SYN[v]) for v in SIZES},
        "color": {v: pick(generator.COLOR_SYN.get(v, [v])) for v in COLORS},
        "shape": {v: pick(generator.SHAPE_SYN[v]) for v in SHAPES},
        # One word per material, valid in both the adjective slot ("a rubber
        # cube") and the noun slot ("made of rubber"); otherwise moving an
        # object between phrase styles would change "rubber" into "rubbery".
        "material": {v: pick([w for w in generator.MATERIAL_ADJ[v] if w in generator.MATERIAL_NOUN[v]])
                     for v in MATERIALS},
        "slot_style": [rng.randrange(N_OBJECT_STYLES) for _ in range(n_objects)],
        "slot_made": [rng.choice(["made of", "built from"]) for _ in range(n_objects)],
        "mention_style": [[rng.randrange(N_OBJECT_STYLES) for _ in range(2)] for _ in range(n_relations)],
        "mention_made": [[rng.choice(["made of", "built from"]) for _ in range(2)] for _ in range(n_relations)],
        "relation_phrase": [rng.random() for _ in range(n_relations)],
        "relation_adverb": [rng.random() for _ in range(n_relations)],
        "relation_opener": [rng.randrange(len(generator.RELATION_OPENERS)) for _ in range(n_relations)],
        "opener": rng.randrange(len(generator.OPENERS)),
        "use_digit": rng.random() < 0.12,
        "tail_joiner": rng.choice(["and ", "plus "]),
    }
    if lexicon is not None:
        for key in LEXICON_KEYS:
            plan[key] = dict(lexicon[key])
    return plan


def object_words(obj: dict, plan: dict, style: int, made: str) -> str:
    size = plan["size"][obj["size"]]
    color = plan["color"][obj["color"]]
    shape = plan["shape"][obj["shape"]]
    adj = noun = plan["material"][obj["material"]]
    # Same six formats, in the same order, as generator.OBJECT_STYLES.
    return (
        f"{size} {color} {adj} {shape}",
        f"{color} {adj} {shape} that looks {size}",
        f"{shape} {made} {noun}, {color} and {size}",
        f"{adj} {shape} in {color}, on the {size} side",
        f"{color} {size} {shape} with a {adj} finish",
        f"{shape} that is {size}, {color}, and {adj}",
    )[style]


def render(generator, world: dict, plan: dict, object_order: list[int], relation_order: list[int], pattern: int) -> str:
    objects = world["objects"]
    phrases = []
    for slot, index in enumerate(object_order):
        words = object_words(objects[index], plan, plan["slot_style"][slot], plan["slot_made"][slot])
        phrases.append(f"{generator.art(words)} {words}")
    sep = "; " if any("," in p or "--" in p for p in phrases) else ", "
    joined = sep.join(phrases[:-1]) + sep + plan["tail_joiner"] + phrases[-1]
    count = str(len(objects)) if plan["use_digit"] else generator.NUMBER_WORDS.get(len(objects), str(len(objects)))
    inventory = generator.OPENERS[plan["opener"]](count, joined, None)
    relations = []
    # Relation-sentence choices belong to the relation itself, not to where
    # its sentence appears, so reordering sentences does not change words.
    for r_index in relation_order:
        relation = world["relations"][r_index]
        kind = relation["relation"]
        mentions = []
        for role, object_index in enumerate((relation["subject"], relation["anchor"])):
            words = object_words(objects[object_index], plan,
                                 plan["mention_style"][r_index][role], plan["mention_made"][r_index][role])
            mentions.append(f"the {words}")
        pool = generator.RELATION_PHRASES[kind]
        adverbs = generator.COMPARE_ADV[kind]
        phrase = pool[int(plan["relation_phrase"][r_index] * len(pool))]
        adverb = adverbs[int(plan["relation_adverb"][r_index] * len(adverbs))]
        opener = generator.RELATION_OPENERS[plan["relation_opener"][r_index]]
        relations.append(opener(mentions[0], phrase, mentions[1], adverb, None))
    sentences = [inventory] + relations if pattern == 0 else relations + [inventory]
    return " ".join(sentences)


def descriptor(obj: dict) -> tuple:
    return tuple(obj[key] for key in ATTRIBUTES)


def scene_signature(world: dict) -> str:
    objects = world["objects"]
    return json.dumps({
        "objects": sorted(descriptor(o) for o in objects),
        "relations": sorted((r["relation"], descriptor(objects[r["subject"]]), descriptor(objects[r["anchor"]]))
                            for r in world.get("relations", [])),
    })


def has_twins(world: dict) -> bool:
    descriptors = [descriptor(o) for o in world["objects"]]
    return len(descriptors) != len(set(descriptors))


def binding_swap(world: dict, kind: str, rng: random.Random) -> dict | None:
    """Return the world with one binding exchanged, or None if no valid swap exists."""
    objects = world["objects"]
    if kind == "relation":
        relations = world.get("relations", [])
        if not relations:
            return None
        swapped = copy.deepcopy(world)
        relation = swapped["relations"][rng.randrange(len(relations))]
        relation["subject"], relation["anchor"] = relation["anchor"], relation["subject"]
        return swapped
    mentions = Counter()
    for relation in world.get("relations", []):
        mentions[relation["subject"]] += 1
        mentions[relation["anchor"]] += 1
    candidates = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i][kind] == objects[j][kind]:
                continue
            # Relation sentences repeat the descriptions of the objects they
            # mention.  Swapping between objects mentioned a different number
            # of times would change those words, so only equal counts qualify.
            if mentions[i] != mentions[j]:
                continue
            # Objects that differ *only* in this attribute would just trade
            # places: the scene would not change.
            if not any(objects[i][other] != objects[j][other] for other in ATTRIBUTES if other != kind):
                continue
            candidates.append((i, j))
    rng.shuffle(candidates)
    for i, j in candidates:
        swapped = copy.deepcopy(world)
        a, b = swapped["objects"][i], swapped["objects"][j]
        a[kind], b[kind] = b[kind], a[kind]
        if not has_twins(swapped):
            return swapped
    return None


def build_items(generator, tokenizer, manifest: str, num_worlds: int, seed: int, max_tokens: int):
    """Captions plus (query, positive, negative) triples for every condition."""
    worlds = []
    with open(manifest) as handle:
        for line in handle:
            row = json.loads(line)
            if not has_twins(row["world"]) and len(row["world"]["objects"]) >= 3:
                worlds.append(row)
            if len(worlds) == num_worlds:
                break
    captions: list[dict] = []
    index_of: dict[str, int] = {}

    def add(text, world_index, role, world_signature, scene):
        if text not in index_of:
            index_of[text] = len(captions)
            captions.append({"index": len(captions), "caption": text, "world_index": world_index,
                             "role": role, "scene_signature": world_signature, "world": scene})
        return index_of[text]

    def bag(text):
        return Counter(tokenizer.tokenize(text))

    items, stats = [], Counter()
    by_shape = {}
    for world_index, row in enumerate(worlds):
        world = row["world"]
        by_shape.setdefault((len(world["objects"]), len(world.get("relations", []))), []).append(world_index)
    for world_index, row in enumerate(worlds):
        world = row["world"]
        n, m = len(world["objects"]), len(world.get("relations", []))
        rng = random.Random(seed * 1_000_003 + world_index)
        plan = make_plan(generator, rng.randrange(2**31), n, m)
        other_plan = make_plan(generator, rng.randrange(2**31), n, m)
        identity, rel_identity = list(range(n)), list(range(m))
        reorder = identity[1:] + identity[:1]
        signature = scene_signature(world)
        query = render(generator, world, plan, identity, rel_identity, 0)
        positives = {
            "reorder": render(generator, world, plan, reorder, list(reversed(rel_identity)), 0),
            "reword": render(generator, world, other_plan, reorder, rel_identity, 1),
        }
        texts = [query, *positives.values()]
        if any(len(tokenizer.tokenize(t)) > max_tokens for t in texts):
            stats["world_dropped_too_long"] += 1
            continue
        if positives["reorder"] == query or bag(positives["reorder"]) != bag(query):
            stats["world_dropped_reorder_not_lexically_matched"] += 1
            continue
        q = add(query, world_index, "query", signature, world)
        pos = {name: add(text, world_index, f"positive_{name}", signature, world) for name, text in positives.items()}
        negatives = {}
        for kind in SWAP_TYPES:
            swapped = binding_swap(world, kind, rng)
            if swapped is None:
                stats[f"{kind}_no_valid_swap"] += 1
                continue
            text = render(generator, swapped, plan, identity, rel_identity, 0)
            if scene_signature(swapped) == signature or text == query:
                stats[f"{kind}_not_a_different_scene"] += 1
                continue
            if bag(text) != bag(query):
                stats[f"{kind}_token_multiset_changed"] += 1
                continue
            if len(tokenizer.tokenize(text)) > max_tokens:
                stats[f"{kind}_too_long"] += 1
                continue
            negatives[kind] = add(text, world_index, f"negative_{kind}", scene_signature(swapped), swapped)
        # Sanity control: a different world with the same object and relation
        # counts, rendered with the query's plan.  Its values differ, so any
        # functional representation should reject it.
        peers = [w for w in by_shape[(n, m)] if w != world_index]
        if peers:
            other = worlds[rng.choice(peers)]["world"]
            text = render(generator, other, plan, identity, rel_identity, 0)
            if scene_signature(other) != signature and len(tokenizer.tokenize(text)) <= max_tokens:
                negatives["different_scene"] = add(text, world_index, "negative_different_scene", scene_signature(other), other)
        for kind, neg in negatives.items():
            for positive_name, p in pos.items():
                items.append({"world_index": world_index, "negative_type": kind,
                              "positive_type": positive_name, "query": q, "positive": p, "negative": neg})
    return worlds, captions, items, stats


def conjunction_labels(world: dict) -> set[tuple]:
    """Binding-dependent facts about a scene.

    * ``("object", A, a, B, b)``: some object has attribute A = a and B = b.
    * ``("relation", s, kind, t)``: some object of shape s is ``kind`` of some
      object of shape t, with left/behind rewritten as right/front so that
      "X left of Y" and "Y right of X" are the same fact.
    """
    objects = world["objects"]
    labels = set()
    for obj in objects:
        for first in range(len(ATTRIBUTES)):
            for second in range(first + 1, len(ATTRIBUTES)):
                a, b = ATTRIBUTES[first], ATTRIBUTES[second]
                labels.add(("object", a, obj[a], b, obj[b]))
    for relation in world.get("relations", []):
        subject, anchor, kind = objects[relation["subject"]]["shape"], objects[relation["anchor"]]["shape"], relation["relation"]
        if kind == "left":
            subject, anchor, kind = anchor, subject, "right"
        elif kind == "behind":
            subject, anchor, kind = anchor, subject, "front"
        labels.add(("relation", subject, kind, anchor))
    return labels


def all_conjunction_labels() -> list[tuple]:
    values = {"color": COLORS, "shape": SHAPES, "material": MATERIALS, "size": SIZES}
    labels = []
    for first in range(len(ATTRIBUTES)):
        for second in range(first + 1, len(ATTRIBUTES)):
            a, b = ATTRIBUTES[first], ATTRIBUTES[second]
            labels += [("object", a, x, b, y) for x in values[a] for y in values[b]]
    labels += [("relation", s, kind, t) for s in SHAPES for kind in ("right", "front") for t in SHAPES]
    return labels
