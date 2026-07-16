"""External data-provider adapters and deterministic runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from regime_allocation.data.providers.alfred_web import AlfredWebDownloadClient
from regime_allocation.data.providers.fred_api import (
    FredApiConfigurationError,
    FredApiDownloadClient,
)
from regime_allocation.data.providers.vintage_matrix import VintageMatrixProvider


@dataclass(frozen=True)
class ProviderSelection:
    """Non-secret record of requested and resolved acquisition providers."""

    requested: str
    selected: str
    client: VintageMatrixProvider


def select_vintage_provider(
    provider: str = "auto",
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderSelection:
    """Resolve ``auto``, ``fred``, or ``alfred`` without silent failover."""

    aliases = {
        "auto": "auto",
        "fred": "fred",
        "fred_api": "fred",
        "alfred": "alfred",
        "alfred_web": "alfred",
    }
    try:
        requested = aliases[provider.strip().lower()]
    except KeyError:
        raise ValueError(
            "provider must be one of: auto, fred, alfred"
        ) from None

    source = os.environ if environ is None else environ
    api_key = source.get("FRED_API_KEY", "")
    has_api_key = bool(api_key.strip())

    if requested == "fred" or (requested == "auto" and has_api_key):
        if not has_api_key:
            raise FredApiConfigurationError(
                "FRED_API_KEY is required when the FRED provider is selected"
            )
        client: VintageMatrixProvider = FredApiDownloadClient(api_key)
    else:
        client = AlfredWebDownloadClient()

    return ProviderSelection(
        requested=requested,
        selected=client.provider_id,
        client=client,
    )


__all__ = [
    "AlfredWebDownloadClient",
    "FredApiConfigurationError",
    "FredApiDownloadClient",
    "ProviderSelection",
    "VintageMatrixProvider",
    "select_vintage_provider",
]
