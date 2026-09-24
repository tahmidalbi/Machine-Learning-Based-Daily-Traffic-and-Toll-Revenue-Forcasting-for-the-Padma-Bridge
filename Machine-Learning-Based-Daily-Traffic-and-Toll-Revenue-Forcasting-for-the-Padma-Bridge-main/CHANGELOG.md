# Changelog

## 1.0.1 - 2026-09-02

- Fixed Colab/statsmodels compatibility: `VARSummary` is now serialized with
  `str(...)` rather than the unsupported `.as_text()` method.
- Removed the deprecated Granger `verbose` argument while keeping compatibility
  with older statsmodels releases.
- Reset the irregular date index before VAR to remove the irrelevant missing-
  frequency forecast warning.
- Suppressed only the rank-deficient omnibus fixed-effect summary warning in
  the NTL analysis; the DiD coefficient and clustered inference are unchanged.
- Corrected end-to-end stage numbering from 1 through 11.

