# Development during the submission period

The published submission window is August26–October30,2026. This submission presents the RL investigation and recovery work recorded in the project's October2026 artifacts, rather than claiming that every underlying simulator or library was created for the event.

The dated public evidence covers these substantial additions:

- Mechanism inspections and behavioral acceptance criteria, including PPO update-count instrumentation and an end-to-end R1 repair (October16, source seed2071/restart3071).
- Correction-scope validation and the four-term U1 confirmation (October16, source2062/restart3062). This was post-result development with the model loop disabled.
- Residual diagnosis after a first repair: v2 compound runs2120/2121 recovered after two corrections (October17). These are development cases informed by earlier F1/R1 results.
- PPO clipped-surrogate gradient invariants that distinguish a calculation defect even when step counts are normal (October2026, public v5 result).
- A later evidence/action runtime and synchronized normal/F1 render capture (October29), followed by the portable English replay and submission package (October30).

Source anchors preserved in the evidence include R1 `cb60b1cd1597daf2ef8005f9d9a25dea053f68da`, U1 `d392fae74f2ccf2e7d3f06c01c62771c50722e23`, and fresh render `694df50e6c2a196da25b9b40c8788921caebd6a8`. The package's supplied public baseline is `d9f94442adb872e83174672cf84aeb6236906263`. Original file paths and SHA-256 hashes are in `source-map.json`.

The project owner checked Git commit contents and ancestry before final integration. The initial root commit `4acb1ed49db7545060c2425520c93d22d730cceb` is dated `2026-09-26T11:37:21+09:00`. The R1 anchor above is dated `2026-10-16T17:08:58+09:00`, U1 `2026-10-16T10:30:28+09:00`, and fresh render `2026-10-29T08:13:25+09:00`. All four commits exist and are ancestors of the checked HEAD. These checks place the repository root and the three selected implementation anchors within the submission period; they are not an audit of every experiment timestamp. Existing simulator, model, and other upstream components retain their own authorship and licenses.


## Additional October evidence and submission integration

The October 16 NVIDIA-hosted NIM development records compare paired model-request times and retain mixed quality results. The later VL records add policy-frame observations with explicit fixed-evaluation correspondence and separate auxiliary cohorts. The seed-2221 NIM3.5 path connects an actual inspection to a correction proposal, fresh retraining and a not_recovered evaluation result.

The RI219 update adds these already measured facts to the English description and a provider-labeled evidence panel. It preserves the actual eight-record Token Factory replay, its host validation, HOLD behavior and authorization-required inspection export. The 168-second v3 video is retained unchanged. This update adds no model call, GPU run, platform lifecycle action or new performance measurement.
