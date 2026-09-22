"""Immutable raw objects and independently timed, credential-free capture receipts.

Objects deduplicate by SHA-256; each capture creates a distinct receipt. Imported
caches retain an unknown retrieval time instead of borrowing the file mtime.
This module performs no network requests and does not interpret source bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit
from uuid import uuid4


class SnapshotIntegrityError(ValueError):
    """An immutable object, receipt, or storage path failed validation."""


_CREDENTIAL_NAMES = (
    "apikey",
    "accesskey",
    "privatekey",
    "signingkey",
    "sessionkey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "authorization",
    "authentication",
    "signature",
    "sessionid",
    "cookie",
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:api[-_ ]?key|access[-_ ]?key|private[-_ ]?key|signing[-_ ]?key|secret|"
    r"token|password|passwd|authorization|authentication|cookie|credential|"
    r"session[-_ ]?id)\s*[:=]|\b(?:bearer|basic)\s+\S+"
)


def _decoded(value: str) -> str:
    for _ in range(5):
        decoded = unquote(value)
        if decoded == value:
            return value
        value = decoded
    raise ValueError("excessively encoded metadata is not permitted")


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("metadata text must be nonempty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("metadata must not contain control characters")
    return value


def _safe_key(value: str) -> None:
    normalized = re.sub(r"[^a-z0-9]", "", _decoded(_text(value)).lower())
    if normalized in {"key", "pwd", "auth", "sig", "jwt", "pat", "sas"} or any(
        word in normalized for word in _CREDENTIAL_NAMES
    ):
        raise ValueError("credential-like metadata keys are not permitted")


def validate_source_url(value: str) -> str:
    """Reject authenticated URLs rather than silently persisting a partial redaction."""

    value = _text(value)
    decoded = _decoded(value)
    if _CREDENTIAL_ASSIGNMENT.search(decoded):
        raise ValueError("credential-like URL content is not permitted")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme == "https"
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and not parts.fragment
            and parts.port in (None, 443)
        )
    except ValueError:
        raise ValueError("source URL must be a public HTTPS URL") from None
    if not valid or any(character.isspace() for character in value):
        raise ValueError("source URL must be a public HTTPS URL without userinfo or fragment")
    for key, entry in parse_qsl(parts.query, keep_blank_values=True):
        _safe_key(key)
        _validate_metadata(entry)
    return value


def _validate_metadata(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _safe_key(key)
            _validate_metadata(item)
    elif isinstance(value, list):
        for item in value:
            _validate_metadata(item)
    elif isinstance(value, str):
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("metadata must not contain control characters")
        decoded = _decoded(value)
        if _CREDENTIAL_ASSIGNMENT.search(decoded):
            raise ValueError("credential-like metadata values are not permitted")
        if decoded.lower().startswith(("https://", "http://")):
            validate_source_url(value)
    elif (
        value is None
        or isinstance(value, (bool, int))
        or isinstance(value, float)
        and math.isfinite(value)
    ):
        pass
    else:
        raise ValueError("metadata must contain only finite JSON values")


def _canonical(value: dict) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("capture times must be timezone-aware datetimes")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_times(retrieved_at: Any, status: str, ingested_at: str) -> None:
    if status not in {"observed", "unknown_legacy_cache"}:
        raise ValueError("unsupported retrieval_time_status")
    if (status == "unknown_legacy_cache") != (retrieved_at is None):
        raise ValueError("retrieval_time_status disagrees with retrieved_at")
    parsed = []
    for value in (ingested_at, retrieved_at):
        if value is not None:
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError("receipt times must be explicit UTC timestamps")
            try:
                parsed.append(datetime.fromisoformat(value))
            except ValueError:
                raise ValueError("invalid receipt timestamp") from None
    if len(parsed) == 2 and parsed[1] > parsed[0]:
        raise ValueError("retrieval time cannot follow ingestion time")


class SnapshotStore:
    """Store immutable captures beneath an explicitly supplied directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def _path(self, relative: str) -> Path:
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise SnapshotIntegrityError("snapshot path escapes the store")
        for candidate in (path, *path.parents):
            if candidate == self.root:
                break
            if candidate.is_symlink():
                raise SnapshotIntegrityError("symbolic links are not permitted in the store")
        return path

    def _write_once(self, relative: str, content: bytes, *, deduplicate: bool) -> None:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Linking a fully written temporary file publishes without replacement.
                os.link(temporary, path)
            except FileExistsError:
                if not deduplicate or self._path(relative).read_bytes() != content:
                    raise SnapshotIntegrityError("immutable snapshot collision or corruption")
        finally:
            temporary.unlink(missing_ok=True)

    def capture(
        self,
        content: bytes,
        *,
        provider_id: str,
        series_id: str,
        source_url: str,
        query: dict,
        retrieved_at: datetime | None,
        retrieval_time_status: str,
        source_definition: dict,
    ) -> dict:
        """Preserve bytes and return a distinct, self-verifying capture receipt."""

        if not isinstance(content, bytes):
            raise TypeError("snapshot content must be bytes")
        if not isinstance(query, dict) or not isinstance(source_definition, dict):
            raise TypeError("query and source_definition must be JSON objects")
        if not source_definition:
            raise ValueError("source_definition must not be empty")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", _text(provider_id)):
            raise ValueError("provider_id must be a lowercase identifier")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", _text(series_id)):
            raise ValueError("series_id must be an uppercase source identifier")
        _text(source_definition.get("source_definition_version", source_definition.get("version")))
        validate_source_url(source_url)
        _validate_metadata(query)
        _validate_metadata(source_definition)
        ingested_at = _utc(datetime.now(UTC))
        retrieval = None if retrieved_at is None else _utc(retrieved_at)
        _validate_times(retrieval, retrieval_time_status, ingested_at)
        digest = _sha256(content)
        body = {
            "schema_version": 1,
            "receipt_id": uuid4().hex,
            "provider_id": provider_id,
            "series_id": series_id,
            "source_url": source_url,
            "query": query,
            "retrieved_at": retrieval,
            "retrieval_time_status": retrieval_time_status,
            "ingested_at": ingested_at,
            "source_definition": source_definition,
            "content_sha256": digest,
            "content_size_bytes": len(content),
            "object_path": f"objects/sha256/{digest[:2]}/{digest}.bin",
        }
        # Round-trip detaches the receipt from caller-owned mutable metadata.
        receipt = json.loads(_canonical(body))
        receipt["receipt_sha256"] = _sha256(_canonical(body))
        receipt["receipt_path"] = (
            f"receipts/{receipt['receipt_id']}-{receipt['receipt_sha256']}.json"
        )
        self._write_once(receipt["object_path"], content, deduplicate=True)
        self._write_once(receipt["receipt_path"], _canonical(receipt), deduplicate=False)
        return receipt

    def read(self, receipt: dict) -> bytes:
        """Verify both the persisted receipt and raw-object digest before returning bytes."""

        required = {
            "schema_version",
            "receipt_id",
            "provider_id",
            "series_id",
            "source_url",
            "query",
            "retrieved_at",
            "retrieval_time_status",
            "ingested_at",
            "source_definition",
            "content_sha256",
            "content_size_bytes",
            "object_path",
            "receipt_sha256",
            "receipt_path",
        }
        if not isinstance(receipt, dict) or set(receipt) != required:
            raise SnapshotIntegrityError("invalid receipt fields")
        if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
            raise SnapshotIntegrityError("unsupported receipt schema")
        for field, length in (("content_sha256", 64), ("receipt_sha256", 64), ("receipt_id", 32)):
            if not isinstance(receipt[field], str) or not re.fullmatch(
                rf"[0-9a-f]{{{length}}}", receipt[field]
            ):
                raise SnapshotIntegrityError("invalid receipt identifier")
        digest = receipt["content_sha256"]
        expected_object = f"objects/sha256/{digest[:2]}/{digest}.bin"
        expected_receipt = f"receipts/{receipt['receipt_id']}-{receipt['receipt_sha256']}.json"
        if receipt["object_path"] != expected_object or receipt["receipt_path"] != expected_receipt:
            raise SnapshotIntegrityError("receipt paths do not match their identifiers")
        body = {
            key: value
            for key, value in receipt.items()
            if key not in {"receipt_sha256", "receipt_path"}
        }
        if _sha256(_canonical(body)) != receipt["receipt_sha256"]:
            raise SnapshotIntegrityError("receipt metadata hash mismatch")
        _validate_metadata(receipt["query"])
        _validate_metadata(receipt["source_definition"])
        validate_source_url(receipt["source_url"])
        _validate_times(
            receipt["retrieved_at"], receipt["retrieval_time_status"], receipt["ingested_at"]
        )
        if self._path(expected_receipt).read_bytes() != _canonical(receipt):
            raise SnapshotIntegrityError("persisted receipt differs from its declared metadata")
        content = self._path(expected_object).read_bytes()
        if _sha256(content) != digest or len(content) != receipt["content_size_bytes"]:
            raise SnapshotIntegrityError("raw snapshot hash or size mismatch")
        return content
