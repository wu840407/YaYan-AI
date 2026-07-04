# YaYan-AI（雅言）

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C)
![Offline](https://img.shields.io/badge/deployment-100%25_offline-brightgreen)
![Dialects](https://img.shields.io/badge/Chinese_dialects-22-orange)

> **Fully-offline multi-dialect speech intelligence system.**
> Converts speech across **22 Chinese dialects and 40+ world languages** into structured
> Traditional-Chinese transcripts — with speaker diarization, character-level timestamps,
> and an LLM-assisted correction loop. Designed for **air-gapped, privacy-critical
> environments**: no cloud, no API calls, no data leaving the machine.

---

## Why YaYan-AI

Off-the-shelf ASR works for Mandarin and English. It falls apart on **low-resource
Chinese dialects** — Hokkien (Taiwanese), Hakka, Wu, Cantonese variants — exactly the
speech that matters in real-world field recordings, call archives, and oral-history
preservation. And in privacy-critical settings, shipping audio to a cloud API is not
an option at all.

YaYan-AI solves both: state-of-the-art dialect recognition, running entirely on
local GPUs.

## Pipeline Architecture

```mermaid
flowchart LR
    A["🎙️ Audio input<br/>meetings · calls · field recordings"] --> D["pyannote 3.1<br/>speaker diarization (≤5 speakers)"]
    D --> L["Per-segment<br/>language / dialect ID"]
    L -->|"22 Chinese dialects"| C["Dolphin-CN-Dialect<br/>dialect ASR"]
    L -->|"40+ languages"| W["Whisper large-v3<br/>multilingual ASR"]
    C --> Q["Qwen3-14B<br/>correction · normalization<br/>→ Traditional Chinese"]
    W --> Q
    Q --> T["📄 Structured transcript<br/>speaker labels + char-level timestamps"]
    T --> F["✏️ Editable feedback loop<br/>human fixes → LLM re-refinement"]
    F -.-> Q
```

**Design decisions that matter:**
- **Per-segment language ID** — a single conversation can mix Mandarin, Hokkien and
  English; routing each segment independently prevents one dominant language from
  swallowing the others.
- **Two-stage ASR routing** — dialect-specialized model (Dolphin) for Chinese variants,
  Whisper large-v3 for everything else; each model does only what it is best at.
- **LLM correction stage** — Qwen3-14B normalizes ASR output into readable Traditional
  Chinese and fixes dialect-specific transcription artifacts.
- **Human-in-the-loop** — user edits feed back into the correction stage, so accuracy
  improves on *your* audio domain over time.

## Features

| Capability | Detail |
|---|---|
| Chinese dialects | 22 — Mandarin, Cantonese, Hokkien/Taiwanese, Hakka, Wu (Shanghai/Suzhou/Wenzhou), Hunanese, … |
| World languages | 40+ via Whisper large-v3 (JA / KO / EU / SEA / ME) |
| Speaker diarization | Up to 5 speakers (A–E), pyannote 3.1 |
| Timestamps | Character-level |
| Output | Structured Traditional-Chinese transcript, per-speaker |
| Correction | Qwen3-14B two-stage + editable feedback loop |
| Deployment | 100 % offline — air-gapped & privacy-critical environments |
| Interface | Gradio web UI |

## Requirements

| Component | Spec |
|---|---|
| OS | Ubuntu 22.04+ |
| GPU | 2× NVIDIA GPU, ~20 GB VRAM each *<!-- TODO(確認): RTX 4000 Ada ×2 還是 RTX 6000 ×2？照實填 -->* |
| Driver / CUDA | 535+ / CUDA 12.1 |
| Python | 3.10 |
| Disk | ~50 GB (models ≈ 19 GB + workspace) |
| RAM | 32 GB+ |

Core stack: PyTorch 2.3.1 · transformers 4.51.3 · pyannote.audio 3.1 · Gradio 4.44

## Quickstart

```bash
git clone https://github.com/wu840407/YaYan-AI.git
cd YaYan-AI
pip install -r requirements.txt

# Download models (online phase; system runs offline afterwards)
bash scripts/download_models.sh   # <!-- TODO(確認): 對齊實際腳本名 -->

python app.py                     # Gradio UI on http://localhost:7860
```

<!-- TODO(你補): 一張 Gradio 介面截圖或 20 秒 demo GIF，放 docs/img/，效果 > 一千字 -->

## Performance

<!-- TODO(你補,高價值): 一個小 benchmark 表 — 例:
| 測試集 | 方言 | CER/WER | RTF |
台語真實錄音 30 min | Hokkien | xx% | 0.x
沒有數字也可先寫定性描述（1 小時錄音處理時間、單卡/雙卡差異） -->

## Roadmap

- [ ] Fine-tuning on H200 hardware for Taiwanese-Hokkien accuracy (2026 H2)
- [ ] Kafka-based high-throughput batch pipeline
- [ ] Streaming (real-time) mode

## License & Author

<!-- TODO(確認): 選 license，建議 MIT 或 Apache-2.0；模型各自沿用上游授權 -->
Maintained by [ChengRung Wu](https://wu840407.github.io) ([@wu840407](https://github.com/wu840407)).
Questions / issues welcome.

---

## 中文說明（摘要）

**雅言 YaYan-AI** 是全離線的多方言語音辨識系統：22 種漢語方言＋40+ 種語言 → 繁體中文逐字稿，
含語者分離（至多 5 人）、字元級時間戳、Qwen3-14B 校正與可編輯回饋迴路。
專為**離線、隱私敏感環境**設計——不連雲端、資料不出機器。
架構：pyannote 3.1 分離語者 → 逐段語言判定 → Dolphin（方言）/ Whisper large-v3（外語）→ Qwen3-14B 校正輸出。
