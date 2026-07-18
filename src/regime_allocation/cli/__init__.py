"""Command-line entry points for rebuilding Model 01 artifacts.

Each command reads a versioned YAML configuration, delegates calculations to
the package modules, and writes auditable CSV/JSON artifacts below the project
root.  The commands form a staged pipeline—target data, evidence, transition
dynamics, inference, then allocation—so later stages can verify upstream hashes
without embedding model calculations in shell scripts.
"""
