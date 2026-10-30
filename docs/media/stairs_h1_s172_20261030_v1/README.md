# H1 Stair Playback Evidence

Two deterministic Isaac Lab playback clips use the official pretrained H1 rough-terrain checkpoint.

| Clip | Frames | FPS | Duration | Forward distance | Height change | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Stairs up | 625 | 50 | 12.5 s | 2.97664 m | +0.57906 m | completed |
| Stairs down | 625 | 50 | 12.5 s | 2.82690 m | -0.67343 m | completed |

Each clip includes its video, state trace, manifest, exact-frame receipt, and shutdown evidence. `source_manifest.json` binds the checkpoint, runner, runtime, and acceptance boundary. `SHA256SUMS.txt` covers every other public file.

This is an official-checkpoint H1 playback result. It uses no new training or model API calls and is not RL Triage recovery evidence. Go2 results are not included in these counts.
