# Third-party source notices

The TD3 implementation in `racing_lab/rl/td3.py` is a modified version of Benjamin
David Evans and contributors' [F1TENTH benchmarks](https://github.com/BDEvan5/f1tenth_benchmarks),
revision `c9730789dd0fa85fc18dd4b600c5246e134ebe22`, licensed under Apache-2.0.
Changes add configurable dimensions/device, explicit bootstrap discounts, separate RNG,
update counters and safe checkpoint serialization. The complete license is retained
in [TD3_LICENSE.txt](racing_lab/rl/TD3_LICENSE.txt). Unmodified numerical-parity fixtures,
their license and hashes are in `tests/fixtures/td3_author/`. These files are excluded
from formatting. Project-original code is MIT; the distributed package includes both
MIT and Apache-2.0 components.

The SAC implementation in `racing_lab/rl/sac.py`, the control/history adaptation in `racing_lab/rl/environment.py`, and the offline parity fixtures in `tests/fixtures/sac_author/` derive from:

- Adrian Remonda's [Assetto Corsa Gym](https://github.com/dasGringuen/assetto_corsa_gym), revision `e56094890c2f8fe6ef2c969c3ac3404f3acd4eb4`, MIT, copyright 2024 Adrian Remonda.
- Toshiki Watanabe's [discor.pytorch](https://github.com/toshikwa/discor.pytorch), the SAC implementation identified by the racing authors, MIT, copyright 2020 Toshiki Watanabe.

Full copyright notices, permissions and warranty text are retained in [the packaged licenses](racing_lab/rl/LICENSES.txt) and [fixture licenses](tests/fixtures/sac_author/LICENSES.txt). The source files in the fixtures are unmodified; their revision and hashes are recorded in [provenance.json](tests/fixtures/sac_author/provenance.json). Fixture formatting is excluded from repository style tools so it can remain identical to the source.
