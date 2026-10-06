"""Canonical two-axis taxonomy for host-side molecular edits.

Every edit is described by two orthogonal facts:

``site_type``
    Where the edit happens.  ``atom`` is a hydrogen-bearing heavy atom,
    ``bond`` is a directed non-ring single bond, ``linker`` is a chain atom
    between two retained parts, and ``ring`` is a ring-skeleton position.

``change_type``
    What happens there.  ``addition`` forms a new bond, ``deletion`` removes
    the bound side, and ``replacement`` swaps an atom or a bound side.

The canonical operation label is ``"<site_type>:<change_type>"``.  This module
also maps the historical one-dimensional names (``replace_hydrogen``,
``replace_fragment``, ...) onto the axes so old site tables and old run
artifacts stay readable.
"""

from __future__ import annotations

from typing import Any

SITE_TYPES = ("atom", "bond", "linker", "ring")
CHANGE_TYPES = ("addition", "deletion", "replacement")

SITE_TYPE_DESCRIPTION = {
    "atom": "A hydrogen-bearing heavy atom. Only addition and atom-level replacement apply.",
    "bond": "A directed non-ring single bond. Its bound side can be deleted or replaced.",
    "linker": "A chain atom between two retained parts. Its length can be added to or deleted from.",
    "ring": "A ring-skeleton atom. An addition or an element replacement here is reported as a T2 change.",
}

CHANGE_TYPE_DESCRIPTION = {
    "addition": "Form one new bond at an existing site; no bond is cut.",
    "deletion": "Cut one bond and remove the bound side; nothing is attached.",
    "replacement": "Swap one atom's element, or cut one bond and attach a new fragment.",
}

#: Historical operation name -> canonical ``(site_type, change_type)``.
LEGACY_OPERATION_AXES: dict[str, tuple[str, str]] = {
    "substitute": ("atom", "addition"),
    "replace_hydrogen": ("atom", "addition"),
    "element_swap": ("atom", "replacement"),
    "replace_fragment": ("bond", "replacement"),
    "terminal_substituent_swap": ("bond", "replacement"),
    "delete_substituent": ("bond", "deletion"),
    "ring_atom_swap": ("ring", "replacement"),
    "ring_size_edit": ("ring", "replacement"),
    "chain_length_edit": ("linker", "addition"),
}

#: ``(site_type, change_type)`` -> canonical operation label.
OPERATION_LABELS: dict[tuple[str, str], str] = {
    ("atom", "addition"): "atom:addition",
    ("atom", "replacement"): "atom:replacement",
    ("bond", "deletion"): "bond:deletion",
    ("bond", "replacement"): "bond:replacement",
    ("linker", "addition"): "linker:addition",
    ("linker", "deletion"): "linker:deletion",
    ("ring", "addition"): "ring:addition",
    ("ring", "replacement"): "ring:replacement",
}

#: Which change types each site type can legally host.
SITE_ALLOWED_CHANGE_TYPES: dict[str, tuple[str, ...]] = {
    "atom": ("addition", "replacement"),
    "bond": ("deletion", "replacement"),
    "linker": ("addition", "deletion"),
    "ring": ("addition", "replacement"),
}


class EditTaxonomyError(ValueError):
    """Raised when an edit request does not fit the canonical two-axis schema."""


def operation_label(site_type: str, change_type: str) -> str:
    try:
        return OPERATION_LABELS[(site_type, change_type)]
    except KeyError as error:
        raise EditTaxonomyError(
            f"Unsupported edit axis combination: {site_type}:{change_type}"
        ) from error


def axes_for_operation(operation: str) -> tuple[str, str]:
    """Map a canonical or historical operation name onto ``(site_type, change_type)``."""
    if not isinstance(operation, str) or not operation:
        raise EditTaxonomyError(f"Operation must be a non-empty string; got {operation!r}")
    if ":" in operation:
        site_type, change_type = operation.split(":", 1)
        validate_axes(site_type, change_type)
        return site_type, change_type
    try:
        return LEGACY_OPERATION_AXES[operation]
    except KeyError as error:
        raise EditTaxonomyError(f"Unknown edit operation: {operation!r}") from error


def validate_axes(site_type: Any, change_type: Any) -> tuple[str, str]:
    if site_type not in SITE_TYPES:
        raise EditTaxonomyError(
            f"site_type must be one of {SITE_TYPES}; got {site_type!r}"
        )
    if change_type not in CHANGE_TYPES:
        raise EditTaxonomyError(
            f"change_type must be one of {CHANGE_TYPES}; got {change_type!r}"
        )
    allowed = SITE_ALLOWED_CHANGE_TYPES[site_type]
    if change_type not in allowed:
        raise EditTaxonomyError(
            f"Site type {site_type!r} only allows {allowed}; got change_type {change_type!r}"
        )
    return site_type, change_type


def allowed_change_types_for_site(site_type: str) -> list[str]:
    return list(SITE_ALLOWED_CHANGE_TYPES.get(site_type, ()))


def change_types_from_legacy_operations(operations: Any) -> list[str]:
    """Project a legacy ``allowed_operations`` list onto canonical change types."""
    if not isinstance(operations, list):
        return []
    result: list[str] = []
    for operation in operations:
        try:
            _site_type, change_type = axes_for_operation(operation)
        except EditTaxonomyError:
            continue
        if change_type not in result:
            result.append(change_type)
    return result


def infer_axes(transformation: dict[str, Any]) -> tuple[str | None, str | None]:
    """Best-effort axes for a payload that carries only concrete site fields.

    Scripted clients, ablation harnesses and older run artifacts may pass a
    transformation without the explicit axes.  The concrete fields are enough to
    recover the site unambiguously: an element means an atom-level replacement, a
    bond site means a bond edit, and an atom index with a fragment means an
    addition.
    """
    element = transformation.get("element")
    if isinstance(element, str) and element.strip():
        return "atom", "replacement"
    if isinstance(transformation.get("bond_site_id"), str) and transformation.get("bond_site_id"):
        return "bond", "replacement"
    if transformation.get("edit_atom_index") is not None or transformation.get("atom_index") is not None:
        return "atom", "addition"
    return None, None


def normalize_transformation(transformation: dict[str, Any]) -> dict[str, Any]:
    """Fill ``site_type``/``change_type``/``operation`` on a transformation dict in place.

    New requests carry ``site_type`` and ``change_type``.  Requests recovered from an
    older run artifact may only carry a historical ``operation`` or a ``target_type``;
    those are projected onto the axes so no old record becomes unreadable.
    """
    site_type = transformation.get("site_type")
    change_type = transformation.get("change_type")
    if not isinstance(site_type, str):
        target_type = transformation.get("target_type")
        site_type = target_type if target_type in SITE_TYPES else None
    if not isinstance(site_type, str) or not isinstance(change_type, str):
        operation = transformation.get("operation")
        if isinstance(operation, str) and operation:
            legacy_site, legacy_change = axes_for_operation(operation)
            site_type = site_type or legacy_site
            change_type = change_type or legacy_change
    if not isinstance(site_type, str) or not isinstance(change_type, str):
        inferred_site, inferred_change = infer_axes(transformation)
        site_type = site_type or inferred_site
        change_type = change_type or inferred_change
    if not isinstance(site_type, str) or not isinstance(change_type, str):
        raise EditTaxonomyError(
            "A transformation requires site_type and change_type (or a legacy operation)"
        )
    if site_type == "atom" and change_type == "replacement" and transformation.get("ring_member"):
        site_type = "ring"
    validate_axes(site_type, change_type)
    transformation["site_type"] = site_type
    transformation["change_type"] = change_type
    transformation["operation"] = operation_label(site_type, change_type)
    return transformation


def taxonomy_documentation() -> dict[str, Any]:
    """Human/LLM readable description of the edit space, for the dossier."""
    return {
        "schema": "site_type x change_type",
        "site_types": {
            name: {"description": SITE_TYPE_DESCRIPTION[name],
                   "allowed_change_types": list(SITE_ALLOWED_CHANGE_TYPES[name])}
            for name in SITE_TYPES
        },
        "change_types": {
            name: CHANGE_TYPE_DESCRIPTION[name] for name in CHANGE_TYPES
        },
        "operation_labels": [
            operation_label(site_type, change_type)
            for site_type in SITE_TYPES
            for change_type in SITE_ALLOWED_CHANGE_TYPES[site_type]
        ],
        "rule": (
            "Choose a host-listed site, then choose one change_type that the site lists in "
            "allowed_change_types. The host validates the concrete edit and builds the product."
        ),
    }
