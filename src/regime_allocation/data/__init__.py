"""Point-in-time macroeconomic data acquisition and normalization.

The data layer converts provider-specific FRED or ALFRED responses into a
common observation-by-vintage representation and selects the first appearance
of each observation without consulting later revisions.  It supplies released
levels and release timestamps to feature construction; it does not classify
regimes or estimate model parameters.
"""
