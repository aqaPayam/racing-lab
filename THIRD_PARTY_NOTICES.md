# Third-party source notices

The SAC implementation in `racing_lab/rl/sac.py`, the control/history adaptation in `racing_lab/rl/environment.py`, and the offline parity fixtures in `tests/fixtures/sac_author/` derive from:

- Adrian Remonda's [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym), revision `e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4`, MIT, copyright 2024 Adrian Remonda.
- Toshiki Watanabe's [discor.pytorch](https://github.com/toshikwa/discor.pytorch), the SAC implementation identified by the racing authors, MIT, copyright 2020 Toshiki Watanabe.

Full copyright notices, permissions and warranty text are retained in [the packaged licenses](racing_lab/rl/LICENSES.txt) and [fixture licenses](tests/fixtures/sac_author/LICENSES.txt). The source files in the fixtures are unmodified; their revision and hashes are recorded in [provenance.json](tests/fixtures/sac_author/provenance.json). Fixture formatting is excluded from repository style tools so it can remain identical to the source.
