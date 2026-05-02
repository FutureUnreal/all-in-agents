from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None  # type: ignore[assignment]
    _HAS_JSONSCHEMA = False


@dataclass(frozen=True)
class ArtifactSpec:
    """A required run artifact.

    ``path`` is resolved relative to the run workspace unless it is absolute.
    Absolute paths are allowed only when they remain under the workspace root.
    ``kind`` can be "file", "dir", or "json".
    """

    path: str
    kind: str = "file"
    min_bytes: int = 1
    json_schema: dict[str, Any] | None = None
    description: str = ""


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    ok: bool
    reason: str = ""
    kind: str = "file"


@dataclass(frozen=True)
class ArtifactValidationResult:
    ok: bool
    checks: tuple[ArtifactCheck, ...] = ()

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(c.reason for c in self.checks if not c.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"path": c.path, "ok": c.ok, "reason": c.reason, "kind": c.kind}
                for c in self.checks
            ],
        }


Validator = Callable[[Path], str | None]


@dataclass(frozen=True)
class ArtifactContract:
    """Machine-checkable artifact requirements for a run."""

    artifacts: tuple[ArtifactSpec, ...] = ()
    validators: tuple[Validator, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "validators", tuple(self.validators))

    @classmethod
    def files(cls, *paths: str) -> "ArtifactContract":
        return cls(tuple(ArtifactSpec(path=p) for p in paths))

    @classmethod
    def json_files(cls, schemas: dict[str, dict[str, Any] | None]) -> "ArtifactContract":
        return cls(tuple(
            ArtifactSpec(path=path, kind="json", json_schema=schema)
            for path, schema in schemas.items()
        ))

    def validate(self, workspace_root: str | Path | None = None) -> ArtifactValidationResult:
        root = Path(workspace_root or ".").resolve()
        checks: list[ArtifactCheck] = []

        for spec in self.artifacts:
            checks.append(_validate_spec(root, spec))

        for validator in self.validators:
            try:
                reason = validator(root)
            except Exception as exc:
                reason = f"custom validator failed: {type(exc).__name__}: {exc}"
            checks.append(ArtifactCheck(path=str(root), ok=reason is None, reason=reason or "", kind="custom"))

        return ArtifactValidationResult(
            ok=all(c.ok for c in checks),
            checks=tuple(checks),
        )


def _resolve_under_root(root: Path, artifact_path: str) -> tuple[Path, str | None]:
    raw = Path(artifact_path)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return path, f"artifact path escapes workspace: {artifact_path}"
    return path, None


def _validate_spec(root: Path, spec: ArtifactSpec) -> ArtifactCheck:
    path, error = _resolve_under_root(root, spec.path)
    if error:
        return ArtifactCheck(spec.path, False, error, spec.kind)

    if spec.kind == "dir":
        if not path.is_dir():
            return ArtifactCheck(spec.path, False, "directory is missing", spec.kind)
        return ArtifactCheck(spec.path, True, kind=spec.kind)

    if not path.is_file():
        return ArtifactCheck(spec.path, False, "file is missing", spec.kind)

    if spec.min_bytes > 0 and path.stat().st_size < spec.min_bytes:
        return ArtifactCheck(
            spec.path,
            False,
            f"file is smaller than min_bytes={spec.min_bytes}",
            spec.kind,
        )

    if spec.kind == "json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return ArtifactCheck(spec.path, False, f"invalid JSON: {exc}", spec.kind)
        if spec.json_schema:
            if not _HAS_JSONSCHEMA:
                return ArtifactCheck(
                    spec.path,
                    False,
                    "jsonschema extra is required for schema validation",
                    spec.kind,
                )
            try:
                jsonschema.validate(data, spec.json_schema)  # type: ignore[union-attr]
            except Exception as exc:
                return ArtifactCheck(spec.path, False, f"JSON schema validation failed: {exc}", spec.kind)

    elif spec.kind != "file":
        return ArtifactCheck(spec.path, False, f"unknown artifact kind: {spec.kind}", spec.kind)

    return ArtifactCheck(spec.path, True, kind=spec.kind)
