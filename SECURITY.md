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

