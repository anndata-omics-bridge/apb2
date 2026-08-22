"""The import law and the placement rules, checked against the source rather than described.

Import Linter enforces the direction of every edge; these tests assert the properties a
layers contract cannot phrase — which module is allowed to know two things at once, that no
runtime module compares a declaration discriminator, and that package markers stay empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PARSER_V2 = Path("src/apb2/parserV2")
PARSE_QUANT = PARSER_V2 / "parse_quant"
RULES = PARSER_V2 / "vendor_parse_rules"
RULE_SCHEMA = RULES / "schema"


def _modules(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py")))


def _imported_modules(path: Path) -> frozenset[str]:
    """Every module one file imports, by dotted name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return frozenset(names)


def _root_modules() -> tuple[Path, ...]:
    return tuple(sorted(path for path in PARSER_V2.glob("*.py") if path.name != "__init__.py"))


@pytest.mark.parametrize(
    "path", _modules(PARSER_V2), ids=lambda path: str(path.relative_to(PARSER_V2))
)
def test_no_module_under_parser_v2_reaches_the_search_parameter_model(path: Path) -> None:
    imported = _imported_modules(path)

    assert not any(name.startswith("apb2.vendor_params") for name in imported)


@pytest.mark.parametrize(
    "path", _modules(PARSE_QUANT), ids=lambda path: str(path.relative_to(PARSE_QUANT))
)
def test_the_parse_package_never_imports_the_rule_package_or_its_parent(path: Path) -> None:
    imported = _imported_modules(path)
    root_names = {f"apb2.parserV2.{module.stem}" for module in _root_modules()}

    assert not any(name.startswith("apb2.parserV2.vendor_parse_rules") for name in imported)
    assert not (imported & root_names)


@pytest.mark.parametrize("path", _modules(RULES), ids=lambda path: str(path.relative_to(RULES)))
def test_the_rule_package_imports_nothing_above_itself(path: Path) -> None:
    imported = _imported_modules(path)

    assert not any(name.startswith("apb2.parserV2.parse_quant") for name in imported)
    assert not any(
        name.startswith("apb2.") and not name.startswith("apb2.parserV2.vendor_parse_rules")
        for name in imported
    )


@pytest.mark.parametrize(
    "path", _modules(RULE_SCHEMA), ids=lambda path: str(path.relative_to(RULE_SCHEMA))
)
def test_the_rule_schema_never_imports_its_parent_document_or_loader(path: Path) -> None:
    imported = _imported_modules(path)

    assert "apb2.parserV2.vendor_parse_rules.document" not in imported
    assert "apb2.parserV2.vendor_parse_rules.loader" not in imported


def test_only_a_parent_module_knows_both_children() -> None:
    """Cross-child composition is exactly what a parent-level module is for."""
    both: set[str] = set()
    for path in _modules(PARSER_V2):
        imported = _imported_modules(path)
        rules = any(name.startswith("apb2.parserV2.vendor_parse_rules") for name in imported)
        parse = any(name.startswith("apb2.parserV2.parse_quant") for name in imported)
        if rules and parse:
            both.add(str(path.relative_to(PARSER_V2)))

    assert both <= {"parse_rule_facade.py", "compile.py"}


def test_every_package_marker_stays_empty() -> None:
    markers = sorted(PARSER_V2.rglob("__init__.py"))

    assert markers
    for marker in markers:
        assert marker.read_text(encoding="utf-8").strip() == "", marker


_RUNTIME_MODULES = (
    PARSE_QUANT / "parser.py",
    PARSE_QUANT / "decomposition.py",
    PARSE_QUANT / "fragments.py",
    PARSE_QUANT / "axis_columns.py",
    PARSE_QUANT / "duplicates.py",
    PARSE_QUANT / "modifications.py",
)
"""The modules that run per parse. None of them may ask what kind of thing it is."""

_DISCRIMINATORS = (
    "kind",
    "how",
    "mode",
    "shape",
    "label_strategy",
    "encoding_mode",
    "parser",
    "software_name",
    "quantification_level",
)


def _string_literals(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _attribute_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))


@pytest.mark.parametrize(
    "path", _RUNTIME_MODULES, ids=lambda path: str(path.relative_to(PARSE_QUANT))
)
def test_no_runtime_module_reads_a_declaration_discriminator(path: Path) -> None:
    """Every tag was consumed at construction; a strategy that reads one kept it."""
    read = _attribute_names(path)

    assert not set(_DISCRIMINATORS) & read


@pytest.mark.parametrize(
    "path", _RUNTIME_MODULES, ids=lambda path: str(path.relative_to(PARSE_QUANT))
)
def test_no_runtime_module_compares_a_vendor_or_a_level(path: Path) -> None:
    literals = _string_literals(path)
    vendors = {
        "AlphaDIA",
        "AlphaPept",
        "DIA-NN",
        "FragPipe",
        "MaxQuant",
        "PEAKS",
        "Sage",
        "Spectronaut",
        "WOMBAT",
    }
    levels = {"ion", "peptidoform", "peptide", "protein", "fragment"}

    assert not vendors & literals
    assert not levels & literals


_TAG_VOCABULARY = frozenset(
    {
        "long",
        "wide",
        "delimited_fragment",
        "positional",
        "column",
        "token_regex",
        "site_list",
        "error",
        "keep_first",
        "aggregate",
        "null_only",
        "plain_numeric",
        "regex_numeric",
        "factor",
        "string",
        "integer",
        "number",
        "boolean",
        "coalesce",
        "join_nonempty",
        "stripped_sequence",
        "proforma_sequence",
        "proforma_ion",
        "proforma_fragment",
    }
)
"""Every declarative tag schema 0.3 can write. A table keyed by these selects behaviour."""


def _tag_keyed_tables(path: Path) -> list[str]:
    """Dict literals whose keys are declaration tags — that is, behaviour registries."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or not node.keys:
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        if len(keys) == len(node.keys) and set(keys) <= _TAG_VOCABULARY:
            found.append(", ".join(keys))
    return found


def test_the_registries_live_only_in_the_composition_root() -> None:
    """A runtime module holding a tag table would be choosing behaviour per parse."""
    for path in _modules(PARSER_V2):
        tables = _tag_keyed_tables(path)
        if path.name == "compile.py":
            assert tables, "the composition root is where the tag tables belong"
            continue
        assert not tables, f"{path} holds a tag table: {tables}"


def test_only_the_annData_adapter_reaches_for_a_storage_backend() -> None:
    for path in _modules(PARSE_QUANT):
        imported = _imported_modules(path)
        backends = {"pandas", "numpy", "anndata"} & {name.split(".")[0] for name in imported}
        expected = {"pandas", "numpy", "anndata"} if path.name == "anndata_writer.py" else set()
        assert backends <= expected, f"{path} imports {backends}"


def test_no_construction_function_merely_forwards_its_argument() -> None:
    """A ``make_*`` that only calls one constructor with one argument is a wrapper."""
    for path in _modules(PARSER_V2):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not (node.name.startswith("make_") or node.name.endswith("_for")):
                continue
            body = [statement for statement in node.body if not _is_docstring(statement)]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                continue
            returned = body[0].value
            if not isinstance(returned, ast.Call):
                continue
            forwards = [
                argument
                for argument in returned.args
                if isinstance(argument, ast.Name)
                and argument.id in {parameter.arg for parameter in node.args.args}
            ]
            assert not (len(returned.args) == 1 and len(forwards) == 1 and not returned.keywords), (
                f"{path}:{node.name} forwards its only argument"
            )


def _is_docstring(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def test_no_removed_concept_reappeared() -> None:
    """The names the architecture deleted, checked against the tree rather than remembered."""
    forbidden = (
        "ParsedData",
        "AxisJoinMap",
        "SourceComposer",
        "SourcePacker",
        "ReconstructionTrace",
        "PhysicalCellLedger",
        "LexicalEnvelope",
        "ResolvedCell",
        "ScatterAssembly",
        "ServiceLocator",
        "Builder",
        "AbstractFactory",
    )
    for path in _modules(PARSER_V2):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in source, f"{path} mentions {name}"
