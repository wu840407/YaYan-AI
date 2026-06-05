# CLAUDE.md — YaYan-AI 開發脈絡

> 給 Claude Code 的專案說明。請先讀完再動 code。

## 專案概要

YaYan-AI 是一套**離線部署**的多語言電話錄音轉錄 + 翻譯系統，部署在軍政府內網
（`yayan-server.yn.example.local`）。功能：上傳音檔 → ASR 轉文字 →
LLM 翻譯成台灣正體中文，附說話人分離（A方/B方...）與字級時間戳。

## 硬體 / 環境（不可改變的約束）

- **GPU**: 2× Quadro RTX 6000（Turing 架構 sm_75，各 24GB）
  - ASR 在 cuda:0，LLM 在 cuda:1
- **OS**: Ubuntu 24.04, kernel 鎖在 6.8.0-106（已 apt-mark hold，勿升級）
- **conda env**: `yayan`（Python 3.10）
- **離線**: 正式環境設 `HF_HUB_OFFLINE=1` 等，模型都在 `/data/ai_models/`

## ⚠️ Turing 架構的硬限制（踩過的雷，勿重蹈）

1. **不能用 vLLM**：vLLM 新版要 triton 3.0+，但 Turing + PyTorch 2.3.1 鎖 triton 2.3.1
2. **不能用 AWQ**：autoawq 需要 triton 3.0 的 `tl.interleave`，Turing 跑不動
3. **LLM 固定用 transformers + bitsandbytes NF4**：`Qwen/Qwen3-14B` BF16
   載入時做 4-bit NF4 量化。**不要嘗試換 vLLM 或 AWQ，會浪費好幾天**
4. **版本鎖**：torch==2.3.1+cu121, triton==2.3.1, transformers==4.51.3,
   huggingface_hub==0.30.2（用 `huggingface-cli` 不是 `hf`）

## 技術棧

| 層 | 模型 | 備註 |
|---|---|---|
| ASR 中文方言 | Dolphin-CN-Dialect-Small (`small.cn.pt`) | 22 方言 + 字級時間戳 |
| ASR 全球 | Whisper-large-v3 | 日韓歐美中東東南亞 |
| LID | VoxLingua107 ECAPA (speechbrain) | v4.7 要加 Whisper LID ensemble |
| VAD | silero-vad 6.x | pip-bundled，無需下載 |
| Diarize | pyannote 3.1 | 離線 YAML patch |
| LLM | Qwen3-14B + NF4 | 翻譯與潤飾 |

## Dolphin region codes（重要，別猜）

Dolphin 用自己的命名，**不是 ISO**。正確對照（從 Dolphin/languages.md）：
- `cmn-sw` → `(zh, SICHUAN)`  四川話
- `yue` → `(ct, NULL)`  粵語的 lang_sym 是 **ct** 不是 yue！
- `nan` → `(zh, MINNAN)`  閩南語
- `cdo` → `(zh, FUJIAN)`  福州話/閩東
- `wuu` → `(zh, WU)`，`wuu-wz` → `(zh, WENZHOU)`
詳見 `yayan/asr/dolphin.py` 的 `ROUTING_TO_DOLPHIN`。

## 目前進度（v4.6.0 已上線）

✅ 22 方言 ASR、ABCDE 說話人、`[A方 00:01-00:05]` 時間戳
✅ 分批翻譯（30 行/批，解決長音檔 LLM 截斷）
✅ systemd + Nginx HTTPS + AD GPO 自簽 CA
✅ Dolphin region codes 修正

## v4.7 目標（進行中）— 解「鄉音 + 混合語音翻譯爛」

使用者回報：**濃厚鄉音、混合語音的翻譯品質差**。根因分兩塊：
1. **鄉音爛 = ASR 層**：Dolphin 對重口音識別本身就抓錯字
2. **混合語音爛 = LID 路由**：逐段 LID 判錯語言 → 送錯 ASR 模型

### v4.7-A：滑動窗口 LID
逐段 LID 時借前後各 1.5 秒 context（短段信心提升）。
改 `yayan/pipeline.py` 的 `_slice_with_context`。

### v4.7-B：LID Ensemble
VoxLingua107 + Whisper LID 投票。新增 `yayan/lid_whisper.py`，
用既有 Whisper-large-v3（不額外吃 VRAM）。config `enable_lid_ensemble` 開關。
Whisper 對中文方言區分比 VoxLingua107 強。

### v4.7-C（暫緩，需評估）
台語專用 ASR（`luigisaetta/whisper-medium-zh-tw`）。
**先測 Dolphin nan 在真實台灣台語的表現，不夠好再加**。

## 開發守則

1. **動 code 前先 establish baseline**：用同段音檔跑現版，存 `/data/yayan_baselines/`
2. **改一個測一個**，每次跟 baseline 對照，別一次改三個
3. **不要碰 LLM backend**（vLLM/AWQ 是死路，見上面 Turing 限制）
4. **commit 訊息寫清楚**，這是 career portfolio 專案
5. 測試指令：`bash scripts/start_server.sh`（會先跑 verify_models.py）
6. 正式服務：`sudo systemctl restart yayan`

## GitHub

`https://github.com/wu840407/YaYan_AI`（branch: master，v4.7 開在 v4.7-dev）
