import base64
import datetime as dt
import hashlib
import io
import logging
import os
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

log = logging.getLogger(__name__)

VISION_PROMPT = (
    'You are extracting content from an image embedded in a PDF document. '
    'Describe what the image shows in 2-6 sentences so it reads naturally where it appears in the text. '
    'If it contains text, diagrams, charts, screenshots, or scanned pages, transcribe the important text '
    'and state the key takeaway. '
    'If the image is purely decorative or carries no information, reply with exactly NO_CONTENT.'
)

TRANSCRIBE_PROMPT = (
    'This image is a full scanned page from a document. Transcribe ALL visible text verbatim, '
    'preserving reading order, headings, list items, and table structure (use markdown for tables). '
    'For figures or diagrams inside the page, add a brief bracketed description in place. '
    'Do not summarize, comment, or omit anything. If the page is blank or unreadable, reply with exactly NO_CONTENT.'
)


class PDFLoader(BaseLoader):
    def __init__(self, file_path, *, extract_images=False, mode='page'):
        if mode not in ('single', 'page'):
            raise ValueError("PDF mode must be 'single' or 'page'")
        self.file_path = str(Path(file_path).expanduser())
        self.extract_images = extract_images
        self.mode = mode
        self.ocr = None
        self._vision_client = None
        self.vision_model = os.getenv('PDF_VISION_MODEL', '').strip()
        self.vision_min_size = int(os.getenv('PDF_VISION_MIN_SIZE', '128'))
        self.vision_max_dim = int(os.getenv('PDF_VISION_MAX_DIM', '1568'))
        self.vision_max_tokens = int(os.getenv('PDF_VISION_MAX_TOKENS', '1024'))
        self.vision_fullpage_tokens = int(os.getenv('PDF_VISION_FULLPAGE_MAX_TOKENS', '8192'))
        self.vision_fullpage_ratio = float(os.getenv('PDF_VISION_FULLPAGE_RATIO', '0.6'))
        self.vision_fullpage_max_dim = int(os.getenv('PDF_VISION_FULLPAGE_MAX_DIM', '1568'))
        self.vision_page_text_min = int(os.getenv('PDF_VISION_PAGE_TEXT_MIN', '200'))
        self.vision_concurrency = max(1, int(os.getenv('PDF_VISION_CONCURRENCY', '3')))
        self.vision_disable_thinking = os.getenv('PDF_VISION_DISABLE_THINKING', 'true').lower() != 'false'
        self.cache_dir = Path(os.getenv('PDF_VISION_CACHE_DIR', '/app/backend/data/cache/pdf_vision'))

    def lazy_load(self):
        if self.vision_model:
            try:
                import fitz

                probe = fitz.open(self.file_path)
                probe.close()
            except Exception as e:
                log.warning('PyMuPDF unavailable for %s (%s); falling back to pypdf text extraction.', self.file_path, e)
            else:
                yield from self._lazy_load_vision()
                return
        yield from self._lazy_load_pypdf()

    def _lazy_load_vision(self):
        import fitz

        doc = fitz.open(self.file_path)
        try:
            metadata = {'producer': 'PyMuPDF', 'creator': '', 'creationdate': ''}
            meta = doc.metadata or {}
            for key in ('title', 'author', 'subject', 'keywords', 'creator', 'producer'):
                value = meta.get(key)
                if value:
                    metadata[key] = str(value).strip()
            for src, dst in (('creationDate', 'creationdate'), ('modDate', 'moddate')):
                value = meta.get(src)
                if value:
                    metadata[dst] = str(value).strip()
            metadata.update(source=self.file_path, total_pages=doc.page_count)
            page_count = doc.page_count
            labels = []
            for index in range(page_count):
                try:
                    labels.append(doc.load_page(index).get_label())
                except Exception:
                    labels.append(str(index))
        finally:
            doc.close()

        def work(index):
            # fitz Documents are not thread-safe; each worker opens its own.
            worker_doc = fitz.open(self.file_path)
            try:
                return self._page_text_with_vision(worker_doc.load_page(index))
            finally:
                worker_doc.close()

        if self.vision_concurrency > 1 and page_count > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(self.vision_concurrency, page_count)) as pool:
                texts = list(pool.map(work, range(page_count)))
        else:
            texts = [work(index) for index in range(page_count)]

        if self.mode == 'page':
            for index, text in enumerate(texts):
                yield Document(page_content=text, metadata={**metadata, 'page': index, 'page_label': labels[index]})
        else:
            yield Document(page_content='\n\f'.join(texts), metadata=metadata)

    def _page_text_with_vision(self, page):
        page_area = abs(page.rect) or 1
        parts = []
        text_len = 0
        for block in page.get_text('dict')['blocks']:
            if block['type'] == 0:
                lines = [''.join(span['text'] for span in line['spans']) for line in block.get('lines', [])]
                text = '\n'.join(filter(None, lines)).strip()
                if text:
                    text_len += len(text)
                    parts.append(text)
            elif block['type'] == 1:
                parts.append(block)
        # No usable text layer: the page may be a scan, image slices, or vector-drawn text
        # (print-to-PDF). Transcribe the rendered page once. Blank/photo pages answer
        # NO_CONTENT and fall through to per-block handling.
        if text_len < self.vision_page_text_min:
            transcription = self._transcribe_page(page)
            if transcription:
                return transcription
        out = []
        for part in parts:
            if isinstance(part, str):
                out.append(part)
                continue
            bbox = part.get('bbox')
            fullpage = bool(bbox) and abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / page_area >= self.vision_fullpage_ratio
            description = self._describe_image(part, fullpage=fullpage)
            if description:
                out.append(f'[Image: {description}]')
        return '\n\n'.join(out).strip()

    def _transcribe_page(self, page):
        import fitz

        zoom = min(self.vision_fullpage_max_dim / max(page.rect.width, page.rect.height), 2.0)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            if set(pix.samples) == {255}:
                return ''
            png = pix.tobytes('png')
        except Exception as e:
            log.warning('Page render failed: %s', e)
            return ''
        digest = hashlib.sha256(b'page\n' + png).hexdigest()
        cache_file = self.cache_dir / f'{digest}.txt'
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
        description = self._call_vision(png, fullpage=True)
        if description is None:
            return ''
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(description, encoding='utf-8')
            (self.cache_dir / f'{digest}.png').write_bytes(png)
        except OSError as e:
            log.warning('Could not cache PDF page transcription: %s', e)
        return description

    def _describe_image(self, block, *, fullpage=False):
        if block.get('width', 0) < self.vision_min_size or block.get('height', 0) < self.vision_min_size:
            return ''
        image_bytes = block.get('image')
        if not image_bytes:
            return ''
        png = self._normalize_image(image_bytes, fullpage=fullpage)
        if not png:
            return ''
        digest = hashlib.sha256(('fullpage\n' if fullpage else 'describe\n').encode() + png).hexdigest()
        cache_file = self.cache_dir / f'{digest}.txt'
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
        description = self._call_vision(png, fullpage=fullpage)
        if description is None:
            return ''
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(description, encoding='utf-8')
            (self.cache_dir / f'{digest}.png').write_bytes(png)
        except OSError as e:
            log.warning('Could not cache PDF vision result: %s', e)
        return description

    def _normalize_image(self, image_bytes, *, fullpage=False):
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except (UnidentifiedImageError, OSError) as e:
            log.warning('Skipping unreadable PDF image: %s', e)
            return None
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        max_dim = self.vision_fullpage_max_dim if fullpage else self.vision_max_dim
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()

    def _get_vision_client(self):
        if self._vision_client is None:
            from openai import OpenAI

            self._vision_client = OpenAI(
                base_url=os.getenv('PDF_VISION_API_BASE_URL', 'http://127.0.0.1:8080/v1'),
                api_key=os.getenv('PDF_VISION_API_KEY', 'none'),
                timeout=float(os.getenv('PDF_VISION_TIMEOUT', '180')),
            )
        return self._vision_client

    def _call_vision(self, png, *, fullpage=False):
        prompt = TRANSCRIBE_PROMPT if fullpage else VISION_PROMPT
        max_tokens = self.vision_fullpage_tokens if fullpage else self.vision_max_tokens
        try:
            response = self._get_vision_client().chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': prompt},
                            {
                                'type': 'image_url',
                                'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=0,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}} if self.vision_disable_thinking else None,
            )
            description = (response.choices[0].message.content or '').strip()
        except Exception as e:
            log.warning('Vision description failed for PDF image: %s', e)
            return None
        if not description or description.upper() == 'NO_CONTENT':
            return ''
        return description

    def _lazy_load_pypdf(self):
        from pypdf import PdfReader

        with open(self.file_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = {'producer': 'PyPDF', 'creator': 'PyPDF', 'creationdate': ''}
            for key, value in (reader.metadata or {}).items():
                key = key.removeprefix('/').lower()
                value = value if type(value) in (str, int) else str(value)
                if key in ('creationdate', 'moddate') and isinstance(value, str):
                    try:
                        value = dt.datetime.strptime(value.replace("'", ''), 'D:%Y%m%d%H%M%S%z').isoformat()
                    except ValueError:
                        pass
                metadata[key] = (
                    value.strip()
                    if isinstance(value, str) and key not in ('creationdate', 'moddate', 'page_count', 'file_path')
                    else value
                )
            metadata.update(source=self.file_path, total_pages=len(reader.pages))
            labels = reader.page_labels if self.mode == 'page' else None
            texts = []
            for index, page in enumerate(reader.pages):
                text = page.extract_text()
                if self.extract_images:
                    image_text = self._extract_images(page)
                    if image_text:
                        text = self._merge_image_text(text, image_text)
                text = text.strip()
                if self.mode == 'page':
                    yield Document(page_content=text, metadata={**metadata, 'page': index, 'page_label': labels[index]})
                else:
                    texts.append(text)
            if self.mode == 'single':
                yield Document(page_content='\n\f'.join(texts), metadata=metadata)

    @staticmethod
    def _merge_image_text(text, image_text):
        # Insert before the final paragraphs/footer where possible, matching existing chunks.
        position, separator = len(text), '\n\n'
        for _ in range(2):
            for delimiter in ('\n\n\n', '\n\n'):
                found = text.rfind(delimiter, 0, position)
                if found >= 0:
                    position, separator = found, delimiter
                    break
            else:
                break
        return text[:position] + separator + image_text + text[position:]

    def _extract_images(self, page):
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        if '/Resources' not in page or '/XObject' not in page['/Resources']:
            return ''
        texts = []
        xobjects = page['/Resources']['/XObject']
        for name in xobjects:
            try:
                stream = xobjects[name]
                if stream.get('/Subtype') != '/Image':
                    continue
                try:
                    # Encoded images, including CMYK JPEGs, can go straight to Pillow.
                    image = Image.open(io.BytesIO(stream.get_data()))
                except UnidentifiedImageError:
                    image = stream.decode_as_image()
                pixels = np.array(image.convert('RGB'))
            except Exception as e:
                log.warning('Skipping unreadable PDF image %s: %s', name, e)
                continue

            if self.ocr is None:
                from rapidocr import RapidOCR

                self.ocr = RapidOCR()
            result = self.ocr(pixels)
            if result and result.txts:
                texts.append('\n'.join(result.txts).strip())
        return '\n\n' + '\n'.join(filter(None, texts)) + '\n\n' if any(texts) else ''