# NVIDIA Nemotron 공식 출처 원장 — rev2

열람 UTC: **2026-10-30T09:00:01Z**. NVIDIA 공식 Build, API docs, research report, NVIDIA 조직 Hugging Face 카드만 사용했다.

| ID | 제목 / URL | 게시·개정일 | 정확히 지지하는 claim |
|---|---|---|---|
| S1 | [NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | release 2026-03-11 | old 120B/12B, LatentMoE/Mamba-2/MoE/Attention/MTP, 1M context, text-only, MTP/NVFP4. |
| S2 | [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) | 검색 결과 표기: 7 months ago | old 512 total experts/layer, top-k 22, 2 shared-weight MTP layers. PDF full fetch는 timeout이었으나 NVIDIA search result가 해당 표를 반환했다. |
| S3 | [NIM old endpoint](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/build) | last modified 7 months ago | 정확 old ID, Build Parameters=124B, 1M, text input/output. |
| S4 | [NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) | release 2026-10-15 | new 120B/12B, C-RADIO+connector, 512 routed experts/layer, VL, MTP, 1M 및 training/deployment 설명. |
| S5 | [NIM new model reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b) | release 2026-10-15; updated 15 days ago | new NVFP4, 120B/12B, 512 experts/layer, shared context, data/training 및 engine 범위. |
| S6 | [NIM old infer API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b-infer) | updated 3 months ago | old `reasoning_effort` none/low/high, none이 reasoning tokens 비활성, max_tokens 32768, stream field. 이번 열람에서 HTTP 200. |
| S7 | [NIM new infer API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer) | updated 15 days ago | new `reasoning_effort` none/low/high, multimodal content/tool/json schema, max_tokens 131072. |
| S8 | [Nemotron 3.5 Super and Lightning research page](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Super-and-Lightning/) | published 2026-10-14 | 3.5 overview, conditional efficiency statements, release families. |
| S9 | [Nemotron 3.5 Super and Lightning Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf) | document date 2026-10-27 | recursive MTP, MTP boosting, QAT/serving mechanisms, reported benchmark conditions. |
| S10 | [NIM new endpoint](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b) | last modified 14 days ago | 정확 new ID, Build Parameters=124B, hosted context 262K, multimodal API example. |

## 날짜·표기 주의

- 3.5 research landing page 2026-10-14, model release records 2026-10-15, technical report 2026-10-27을 분리한다.
- Build 124B와 weight-card/API 120B/12B의 격차에 대한 공식 설명은 이번 자료에서 찾지 못했다. encoder 포함 같은 원인을 추론하지 않는다.
