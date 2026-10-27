#!/usr/bin/env python3
"""Produce the candidate worklist for a polymorphism-over-discrimination review.

Runs every mechanical query the skill needs, in the order the skill wants them, and applies
the rejections that need no human judgement. Output is a worklist to read — never a verdict.

Why this exists: each query on its own is a one-liner, but three things make hand-rolling it a
waste of a review's time. (1) The `isinstance` bound needs to know which classes the package
itself defines, which is an AST question, not a grep. (2) The highest-yield query — identity
types that never got a method — needs class bodies inspected, also AST. (3) Getting the order
wrong means reading 200 sites that a 20-line script rejects outright. On one 97-module package
this rejects 57% of isinstance sites before anyone reads anything.

Usage:
    python find_candidates.py <package-dir> [--json]

Sections of the output map 1:1 onto the skill's shapes:

    IDENTITY TYPES WITHOUT METHODS   shape 4  — run first, highest yield
    VALIDATOR MESSAGES               shape 1  — the "X is only valid when Y is Z" tell
    OPTIONAL-FIELD DISCRIMINATION    shape 3  — `.field is None` on an owned type
    MODE-STRING RETURNS              shape 5  — a function returning a literal another branches on
    OWNED-TARGET ISINSTANCE          shape 6  — foreign targets already removed
    WIDE-PARAMETER READS             shape 7  — a queue, not findings
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IDENTITY_PREFIXES = ("Missing", "No", "Non", "Empty", "Absent", "Null", "Unset")

# Shape 1: the tell is in the message text, not the code. "X is only valid when Y is Z" means Y
# is a type discriminator; "X must not exceed Y" is ordinary validation.
VALIDATOR_TELL = re.compile(
    r"is only valid|only valid (if|for|when)|only supported (for|when)"
    r"|cannot define|cannot have|requires .*(when|for)|is not valid for",
    re.IGNORECASE,
)

# Shape 5: literals that name an operation or mode rather than carrying data.
MODE_LITERAL = re.compile(r"^[a-z][a-z0-9_]{1,24}$")

WHOLE_OBJECT_USE = frozenset({
    "model_dump", "model_dump_json", "model_copy", "as_dict", "to_dict",
    "dict", "json", "copy", "model_validate", "to_series",
})


@dataclass
class Hit:
    file: str
    line: int
    detail: str
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "detail": self.detail, **self.extra}


def parse_all(root: Path) -> dict[Path, ast.Module]:
    trees = {}
    for path in sorted(root.rglob("*.py")):
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return trees


def owned_classes(trees: dict[Path, ast.Module]) -> dict[str, tuple[Path, int]]:
    out: dict[str, tuple[Path, int]] = {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                out.setdefault(node.name, (path, node.lineno))
    return out


def isinstance_targets(node: ast.Call) -> list[str]:
    """Class names named in `isinstance(x, <target>)`, flattening `A | B` and tuples."""
    names: list[str] = []
    stack = [node.args[1]]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            names.append(cur.id)
        elif isinstance(cur, ast.Attribute):
            names.append(cur.attr)
        elif isinstance(cur, ast.BinOp):
            stack += [cur.left, cur.right]
        elif isinstance(cur, (ast.Tuple, ast.List)):
            stack += list(cur.elts)
    return names


# --------------------------------------------------------------------------- shape 4

def identity_types(trees, root) -> tuple[list[Hit], list[Hit]]:
    """Absent-case types, split by whether they carry behaviour.

    A codebase that has adopted "replace `| None` with a named absent case" grows these by the
    dozen; nothing in that discipline makes anyone add the method, so the branch survives the
    split. That is why this runs first.
    """
    bare, with_method = [], []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.startswith(IDENTITY_PREFIXES) or node.name.endswith(("Error", "Exception")):
                continue
            methods = [
                b.name for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and not b.name.startswith("_")
            ]
            hit = Hit(str(path.relative_to(root)), node.lineno, node.name,
                      {"methods": methods, "docstring": (ast.get_docstring(node) or "").split("\n")[0]})
            (with_method if methods else bare).append(hit)
    return bare, with_method


def union_members(trees) -> dict[str, set[str]]:
    """Map every class name to the full membership of any union alias it appears in.

    This matters more than it looks. Consumers of an `A | NoA` union almost always discriminate
    on the **present** member — `isinstance(x, GeneName)`, not `isinstance(x, MissingGeneName)` —
    because the interesting branch is the one that has a value to use. Counting sites only
    against the absent member undercounts the shape drastically: on one package it reported 1
    site for a union that is actually discriminated in 6 places across 4 modules. So resolve the
    union first, then count sites against any of its members.
    """
    def members_of(expr: ast.expr) -> set[str]:
        """Names joined by `|`, descending through Annotated[...] and Optional[...]."""
        members: set[str] = set()
        stack = [expr]
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.BitOr):
                stack += [cur.left, cur.right]
            elif isinstance(cur, ast.Name):
                members.add(cur.id)
            elif isinstance(cur, ast.Attribute):
                members.add(cur.attr)
            elif isinstance(cur, ast.Subscript):              # Annotated[A | B, Field(...)]
                stack.append(cur.slice)
            elif isinstance(cur, ast.Tuple):
                stack += list(cur.elts)
            elif isinstance(cur, ast.Constant) and isinstance(cur.value, str):
                try:                                          # a stringified annotation
                    stack.append(ast.parse(cur.value, mode="eval").body)
                except SyntaxError:
                    pass
        return members

    # Unions are declared in more places than a `type X = A | B` alias: many appear only in a
    # return annotation (`-> Parameters | MissingSearchParameters`) or a field type. Harvest all
    # of them, or the membership lookup misses exactly the unions that have no name.
    families: list[set[str]] = []
    for tree in trees.values():
        for node in ast.walk(tree):
            exprs: list[ast.expr] = []
            if hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                exprs.append(node.value)
            elif isinstance(node, ast.Assign) and node.value is not None:
                exprs.append(node.value)
            elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
                exprs.append(node.annotation)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is not None:
                    exprs.append(node.returns)
                exprs += [
                    a.annotation
                    for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs
                    if a.annotation is not None
                ]
            for expr in exprs:
                m = members_of(expr)
                m.discard("None")
                if len(m) >= 2:
                    families.append(m)

    member_to_siblings: dict[str, set[str]] = defaultdict(set)
    for members in families:
        for m in members:
            member_to_siblings[m] |= members
    return member_to_siblings


def discriminator_sites(
    trees, root, siblings: dict[str, set[str]], owned: set[str]
) -> dict[str, list[Hit]]:
    """Where each class — or any sibling in its union — is asked about by `isinstance`.

    Only siblings the package itself defines are counted. A union like `str | MissingText` has a
    foreign member, and every `isinstance(x, str)` in the codebase would otherwise be attributed
    to it — which on one package inflated a 3-site union to 41. Foreign members are bound-M1
    rejects in their own right, so dropping them here is the same rule applied consistently.
    """
    raw: dict[str, list[Hit]] = defaultdict(list)
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "isinstance"
                    and len(node.args) == 2):
                continue
            for name in isinstance_targets(node):
                raw[name].append(Hit(str(path.relative_to(root)), node.lineno, name))

    sites: dict[str, list[Hit]] = {}
    for name in set(raw) | set(siblings):
        family = {m for m in siblings.get(name, {name}) if m in owned} or {name}
        merged = [h for member in family for h in raw.get(member, [])]
        sites[name] = merged
    return sites


# --------------------------------------------------------------------------- shape 1

def validator_messages(trees, root) -> list[Hit]:
    out = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            text = ast.unparse(node.exc)
            if VALIDATOR_TELL.search(text):
                out.append(Hit(str(path.relative_to(root)), node.lineno, text[:150]))
    return out


# --------------------------------------------------------------------------- shape 3

def optional_field_branches(trees, root) -> list[Hit]:
    """`x.field is None` / `is not None` where the arms do different work.

    Reported without judging the arms: distinguishing "different behaviour" from "different
    value" needs reading. Guard clauses that return immediately are filtered out, since those
    are asking what happened.
    """
    out = []
    for path, tree in trees.items():
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            if not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                continue
            if not (isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None):
                continue
            if not isinstance(node.left, ast.Attribute):
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.If):
                body = parent.body
                # a bare guard (`if x.f is None: return` / `raise`) asks what happened → skip
                if len(body) == 1 and isinstance(body[0], (ast.Return, ast.Raise, ast.Continue, ast.Pass)):
                    continue
            out.append(Hit(str(path.relative_to(root)), node.lineno, ast.unparse(node)))
    return out


# --------------------------------------------------------------------------- shape 5

def mode_string_returns(trees, root) -> list[Hit]:
    """Functions returning `-> str` whose returns are bare mode-like literals.

    Only a candidate: the skill's next step is to check whether anything at the far end
    actually branches on the value. If nothing does, it is a label, not a discriminator.
    """
    out = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (node.returns is not None and ast.unparse(node.returns) == "str"):
                continue
            literals = {
                sub.value.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant)
                and isinstance(sub.value.value, str) and MODE_LITERAL.match(sub.value.value)
            }
            if len(literals) >= 2:
                out.append(Hit(str(path.relative_to(root)), node.lineno, node.name,
                               {"literals": sorted(literals)}))
    return out


# --------------------------------------------------------------------------- shape 6

def owned_isinstance(trees, root, owned: set[str]) -> tuple[list[Hit], int, Counter]:
    """Split isinstance sites into candidates and mechanical rejects.

    A class you did not define cannot be given a method, so narrowing it is correct code at a
    parse boundary. That rejection needs no reading, which is why it happens here.
    """
    candidates, rejected = [], 0
    foreign = Counter()
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "isinstance"
                    and len(node.args) == 2):
                continue
            names = isinstance_targets(node)
            if not names:
                continue
            if any(n in owned for n in names):
                candidates.append(Hit(str(path.relative_to(root)), node.lineno, " | ".join(names)))
            else:
                rejected += 1
                foreign.update(names)
    return candidates, rejected, foreign


# --------------------------------------------------------------------------- shape 7

def wide_parameter_reads(trees, root, owned: set[str]) -> list[Hit]:
    """Params typed as an owned class where the body reads exactly one attribute.

    Deliberately a queue and not a finding: the useful move is to re-run shapes 1-6 on the
    attribute being reached for, because a function takes the whole record when the thing it
    actually wants has no type yet.
    """
    out = []
    for path, tree in trees.items():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs:
                if arg.annotation is None:
                    continue
                ann = ast.unparse(arg.annotation).strip("'\"")
                if ann not in owned:
                    continue
                used = {
                    n.attr for n in ast.walk(fn)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == arg.arg
                }
                if len(used) == 1 and not (used & WHOLE_OBJECT_USE):
                    out.append(Hit(str(path.relative_to(root)), fn.lineno, fn.name,
                                   {"param": arg.arg, "type": ann, "reads": sorted(used)[0]}))
    return out


# --------------------------------------------------------------------------- report

def build(root: Path) -> dict:
    trees = parse_all(root)
    owned_map = owned_classes(trees)
    owned = set(owned_map)

    bare, with_method = identity_types(trees, root)
    siblings = union_members(trees)
    sites = discriminator_sites(trees, root, siblings, owned)

    cand_isinstance, rejected, foreign = owned_isinstance(trees, root, owned)

    # Attach consumer counts so the >=2-sites bound can be applied without another pass.
    for h in bare:
        hits = sites.get(h.detail, [])
        family = {m for m in siblings.get(h.detail, {h.detail}) if m in owned} or {h.detail}
        h.extra["consumer_sites"] = len(hits)
        h.extra["consumer_modules"] = len({x.file for x in hits})
        h.extra["union"] = " | ".join(sorted(family)) if len(family) > 1 else "(no union alias found)"

    return {
        "package": str(root),
        "modules": len(trees),
        "classes": len(owned),
        "identity_types_without_methods": [h.as_dict() for h in bare],
        "identity_types_with_methods": [h.as_dict() for h in with_method],
        "validator_messages": [h.as_dict() for h in validator_messages(trees, root)],
        "optional_field_branches": [h.as_dict() for h in optional_field_branches(trees, root)],
        "mode_string_returns": [h.as_dict() for h in mode_string_returns(trees, root)],
        "owned_isinstance": [h.as_dict() for h in cand_isinstance],
        "isinstance_rejected_foreign": rejected,
        "most_checked_foreign": foreign.most_common(8),
        "wide_parameter_reads": [h.as_dict() for h in wide_parameter_reads(trees, root, owned)],
    }


def render(r: dict) -> str:
    L: list[str] = []
    add = L.append
    add(f"{r['package']}: {r['modules']} modules, {r['classes']} classes\n")

    bare = r["identity_types_without_methods"]
    add(f"IDENTITY TYPES WITHOUT METHODS (shape 4) — {len(bare)}   *** run these first ***")
    if not bare:
        add("  none")
    for h in sorted(bare, key=lambda x: -x["consumer_sites"]):
        flag = "FINDING " if h["consumer_sites"] >= 2 else "weak    "
        add(f"  {flag} {h['detail']:32} {h['consumer_sites']:2} sites / "
            f"{h['consumer_modules']} modules   {h['file']}:{h['line']}")
        add(f"           union:     {h['union']}")
        if h["docstring"]:
            add(f"           docstring: {h['docstring'][:88]}")
    add(f"  ({len(r['identity_types_with_methods'])} identity types DO carry methods — those are fine)")
    add("  >=2 consumer sites is the reporting threshold; 1 site is 'two arms that will not grow'.")
    add("  A docstring describing behaviour on a class with no methods IS the finding, in prose.")
    # B2 counts per union, so several sub-threshold types sharing one module each score 'weak'
    # while the module they share is the real finding. Roll them up: the answer is usually the one
    # thing all of them work around, not a method on each.
    clustered = {
        path: n
        for path, n in Counter(h["file"] for h in bare if h["consumer_sites"] < 2).items()
        if n >= 2
    }
    if clustered:
        add("  CLUSTERED WEAK TYPES — below threshold alone, one module together:")
        for path, n in sorted(clustered.items(), key=lambda kv: -kv[1]):
            add(f"    {n} weak identity types in {path}")
        add("    Find the single thing they all work around before adding a method to any of them.")
    add("")

    add(f"VALIDATOR MESSAGES (shape 1) — {len(r['validator_messages'])}")
    for h in r["validator_messages"]:
        add(f"  {h['file']}:{h['line']}  {h['detail']}")
    add("  Keep only those where one field is a kind/mode and the others are its payload.")
    add("  'X must not exceed Y' is validation, not a discriminator.\n")

    ofb = r["optional_field_branches"]
    add(f"OPTIONAL-FIELD DISCRIMINATION (shape 3) — {len(ofb)}  (bare guards already removed)")
    for h in ofb[:40]:
        add(f"  {h['file']}:{h['line']}  {h['detail']}")
    if len(ofb) > 40:
        add(f"  … {len(ofb) - 40} more")
    add("")

    add(f"MODE-STRING RETURNS (shape 5) — {len(r['mode_string_returns'])}")
    for h in r["mode_string_returns"]:
        add(f"  {h['file']}:{h['line']}  {h['detail']}() -> {h['literals']}")
    add("  Now grep each literal: if nothing branches on it, it is a label, not a discriminator.\n")

    oi = r["owned_isinstance"]
    add(f"OWNED-TARGET ISINSTANCE (shape 6) — {len(oi)} candidates, "
        f"{r['isinstance_rejected_foreign']} foreign targets already rejected")
    by_mod = Counter(h["file"] for h in oi)
    for mod, n in by_mod.most_common(12):
        add(f"  {n:4} {mod}")
    add(f"  rejected targets you cannot add a method to: "
        f"{', '.join(n for n, _ in r['most_checked_foreign'])}\n")

    wp = r["wide_parameter_reads"]
    add(f"WIDE-PARAMETER READS (shape 7) — {len(wp)}   *** a queue, not findings ***")
    top = Counter(h["type"] for h in wp).most_common(6)
    for t, n in top:
        add(f"  {n:4} functions take a {t} to read one field")
    add("  For each: re-run shapes 1-6 on the field being reached for. Do not just narrow the")
    add("  signature — a function takes the whole record when the thing it wants has no type yet.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package", type=Path, help="package directory to scan")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of the report")
    args = ap.parse_args()
    if not args.package.is_dir():
        raise SystemExit(f"not a directory: {args.package}")
    report = build(args.package)
    print(json.dumps(report, indent=2) if args.json else render(report))


if __name__ == "__main__":
    main()
