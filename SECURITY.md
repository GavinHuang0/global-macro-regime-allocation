# Credential and data security

This repository is designed so the reproducible research code and public
results do not require secrets.

## Rules

1. Do not place Schwab tokens, refresh tokens, client secrets, API keys, or
   brokerage account identifiers anywhere under the repository root.
2. Keep credentials in an operating-system credential store or a directory
   outside the workspace. Git ignore rules are not an access-control boundary.
3. Do not launch Codex from a shell that exports credential environment
   variables. A sandboxed process may inherit variables even when the backing
   file is outside the repository.
4. A future Schwab adapter must receive credentials through a narrow runtime
   token-provider interface. It must never serialize or log the token object.
5. Store raw account responses outside the repository. Commit only aggregated,
   non-account-specific market data or model outputs whose redistribution is
   permitted.
6. Before publishing, run a secret scanner and inspect Git history—not merely
   the working tree.

## FRED API key

Authenticated data refreshes use the GitHub Actions secret named
`FRED_API_KEY`. The workflow exposes it only to the retrieval step. The Python
provider reads the key from the process environment, validates its documented
format, and does not put it in:

- command-line arguments or subprocesses;
- logs, exception messages, or complete request URLs;
- cache identities, cache metadata, filenames, manifests, or public outputs;
- client representations or source links.

The official FRED API contract places the key in an HTTPS query parameter.
Consequently, the authenticated provider uses an in-process HTTP client and
does not invoke `curl`, whose command line could be visible to other processes.
Transport exceptions are replaced with sanitized project exceptions because a
standard HTTP exception may retain its full request URL.

Automatic provider selection uses FRED when a nonempty key exists and ALFRED
when it does not. An invalid key or failed authenticated request is surfaced;
the process does not silently fail over and obscure acquisition provenance.
The explicit `--provider alfred` mode remains available without credentials.

## Local path example

A reasonable Windows location is:

```text
%USERPROFILE%\.config\global-macro-regime-allocation\
```

That directory should not be added as a workspace root and should not be made
available to Codex. This document intentionally does not define a repository
`.env` workflow.

## Incident response

If a credential is ever committed, revoke and rotate it immediately. Removing
the file in a later commit does not remove it from Git history.
