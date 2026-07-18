"""Provider-independent feature construction for regime models.

The submodules transform already point-in-time-safe observations into monthly
growth/inflation composites and event-level leading evidence.  Inputs must
carry release dates established by the data layer; outputs retain those dates
so downstream training can enforce strict information cutoffs.
"""
