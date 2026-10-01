# owui-pdf-vision

Vision-enabled PDF ingestion for [Open WebUI](https://github.com/open-webui/open-webui).

Out of the box Open WebUI extracts only the text layer of PDFs: images are ignored, and scanned or
print-to-PDF documents (where the text lives inside images or vector paths) come out empty. This
project patches Open WebUI's `PDFLoader` so every page is processed with a local vision model and
the result flows into the normal chunking/embedding/RAG pipeline unchanged.

## How it works

`pdf-vision/pdf.py` is a drop-in replacement for
`backend/open_webui/retrieval/loaders/pdf.py`. Each page is handled in one of three tiers:

1. **Page has a usable text layer** (>= `PDF_VISION_PAGE_TEXT_MIN` chars): the extracted text is
   kept as-is; embedded image blocks get a short vision description spliced in at their layout
   position (`[Image: ...]`).
2. **Page has no text layer** (scans, print-to-PDF of webpages with vector-drawn text): the whole
   page is rendered and transcribed verbatim by the vision model. Blank pages are skipped without
   a model call; pure-photo pages answer `NO_CONTENT` and fall back to tier 1.
3. **Single image covering most of a text-layer page**: transcribed instead of described.

Results are cached on disk keyed by SHA-256 of the (mode + normalized image), so re-ingestion and
duplicate images across documents cost nothing. Pages are transcribed concurrently
(`ThreadPoolExecutor`, one PyMuPDF document per worker). Reasoning is disabled for vision calls
(`chat_template_kwargs.enable_thinking=false`) since transcription is mechanical work.

If PyMuPDF or the vision endpoint is unavailable, the loader falls back to the original
pypdf (+RapidOCR) behavior.

## Requirements

- Open WebUI (this repo derives from `ghcr.io/open-webui/open-webui:main`)
- An OpenAI-compatible endpoint with a vision-capable model (tested with Qwen vision on a local
  runtime; reasoning models work and are handled)

## Deployment

### 1. Build the image

```bash
podman build -t open-webui:vision pdf-vision
# or: docker build -t open-webui:vision pdf-vision
```

The Dockerfile derives from the official image, installs `pymupdf`, and copies in the patched
loader.

### 2. Run

Using the provided `start.sh` (podman, host network):

```bash
./start.sh
```

It generates `.env` with a random `WEBUI_SECRET` on first run, builds the image, and starts the
container. Override the vision settings via `.env` or the environment:

```bash
PDF_VISION_MODEL="Qwen3.8 Flash Next" PDF_VISION_API_BASE_URL="http://127.0.0.1:8080/v1" ./start.sh
```

Or with docker compose:

```bash
docker compose up -d --build
```

### 3. Re-ingest existing PDFs

PDFs ingested before the patch keep their old (text-only) chunks. Re-upload them, or remove and
re-add them to their knowledge bases.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `PDF_VISION_MODEL` | *(empty = disabled)* | Vision model id. Must be set to activate vision. |
| `PDF_VISION_API_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible endpoint. |
| `PDF_VISION_API_KEY` | `none` | API key for the endpoint. |
| `PDF_VISION_TIMEOUT` | `180` | Per-request timeout (seconds). |
| `PDF_VISION_CONCURRENCY` | `3` | Pages transcribed in parallel. |
| `PDF_VISION_DISABLE_THINKING` | `true` | Send `enable_thinking:false` (reasoning models). |
| `PDF_VISION_MAX_TOKENS` | `1024` | Output budget for image descriptions. |
| `PDF_VISION_FULLPAGE_MAX_TOKENS` | `8192` | Output budget for page transcriptions. |
| `PDF_VISION_MAX_DIM` | `1568` | Max image dimension sent for descriptions. |
| `PDF_VISION_FULLPAGE_MAX_DIM` | `1568` | Max rendered-page dimension for transcriptions. |
| `PDF_VISION_FULLPAGE_RATIO` | `0.6` | Image bbox area / page area to count as full-page (tier 3). |
| `PDF_VISION_PAGE_TEXT_MIN` | `200` | Below this extracted-text length a page is treated as a scan. |
| `PDF_VISION_MIN_SIZE` | `128` | Skip images smaller than this (icons, bullets). |
| `PDF_VISION_CACHE_DIR` | `/app/backend/data/cache/pdf_vision` | Description + PNG cache (persists in the Open WebUI data volume). |

## Performance notes

Expect roughly 10s per image-heavy page on a single consumer GPU; the bottleneck is decode
throughput of the vision model. Speedups: disable thinking (default on), raise
`PDF_VISION_CONCURRENCY` (server batching permitting), lower `PDF_VISION_FULLPAGE_MAX_DIM`, or
point ingestion at a dedicated second endpoint so it does not compete with chat traffic.

## Upstream updates

The patch is a single file pinned to the upstream loader's interface. After the base image
(`:main`) updates, rebuild and verify the constructor signature
(`PDFLoader(file_path, extract_images=..., mode=...)`) and `lazy_load()` contract still match.