"""Select the point-in-time data provider without changing model semantics.

The authenticated FRED API is primary when a valid in-memory key is available;
the credential-free ALFRED web adapter remains a reproducible fallback. Both
providers return the contracts in :mod:`vintage_matrix`, allowing acquisition
provenance to differ while first-release selection and downstream calculations
remain identical. Provider-selection records never contain secrets.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

from regime_allocation.data.providers.alfred_web import AlfredWebDownloadClient
from regime_allocation.data.providers.fred_api import (
    FredApiConfigurationError,
    FredApiDownloadClient,
)
from regime_allocation.data.providers.vintage_matrix import (
    DownloadedFirstReleaseObservations,
    FirstReleaseObservation,
    VintageMatrixProvider,
    first_release_observations_from_matrix,
)


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
    "DownloadedFirstReleaseObservations",
    "FirstReleaseObservation",
    "ProviderSelection",
    "VintageMatrixProvider",
    "first_release_observations_from_matrix",
    "select_vintage_provider",
]
