#!/usr/bin/env python3
"""Local workflow server for video processing, transcription, and Gemini cleanup."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from style_options import build_style, public_options

ROOT = Path(__file__).resolve().parent
MEDIA_DIR = ROOT / "media"
MEDIA_DIR.mkdir(exist_ok=True)
JOBS_DIR = ROOT / "jobs"
RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1)
RENDER_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
TRANSCRIPTION_CHUNK_SECONDS = 25
CLEANUP_WORDS_PER_REQUEST = 125
GEMINI_MODEL = "gemini-3.5-flash-lite"
ALIGNMENT_OPTIONS = {"whisperx": "WhisperX", "ctc": "CTC forced aligner"}
WHISPERX_DEVICE = os.environ.get("WHISPERX_DEVICE", "cpu")
_whisperx_align_model = None
_whisperx_metadata = None
CLEANUP_SYSTEM_PROMPT = """You convert Hindi (Devanagari) words into Hinglish written in Roman script, for video captions.

You receive a numbered list of Devanagari words. Return exactly one Roman-script word for every number.

RULES
1. Output exactly one entry per input number. Never merge, split, skip or reorder.
2. Use natural, simple Hinglish spelling as people write it in chat, not academic transliteration. No diacritics.
3. English words that were written in Devanagari must be returned in normal English spelling (मीटिंग -> meeting, बिज़नेस -> business).
4. Use consistent spellings. Follow this style guide every time:
   हैं -> hain, है -> hai, नहीं -> nahi, में (in) -> mein, मैं (I) -> main, क्या -> kya, क्यों -> kyun, और -> aur, तो -> toh, भी -> bhi, को -> ko, का -> ka, की -> ki, के -> ke, से -> se, पर -> par, था/थी/थे -> tha/thi/the, रहा/रही/रहे -> raha/rahi/rahe, अच्छा -> accha, ठीक -> theek
5. Write in lowercase, except proper nouns, brand names, acronyms, and the word "I" if it appears in English.
6. Keep numbers as digits. Do not add punctuation.
7. Words in the GLOSSARY must be spelled exactly as given.
8. Output only valid JSON, with no commentary and no markdown fences.

OUTPUT FORMAT
{"1": "<word>", "2": "<word>", ...}

EXAMPLE
Input:
1:मैं 2:कल 3:मीटिंग 4:में 5:था 6:लेकिन 7:आप 8:नहीं 9:आए
Output:
{"1": "main", "2": "kal", "3": "meeting", "4": "mein", "5": "tha", "6": "lekin", "7": "aap", "8": "nahi", "9": "aaye"}"""


class WorkflowHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json(self, status, payload, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/extract-audio":
            self._extract_audio()
        elif path == "/transcribe":
            self._transcribe_audio()
        elif path == "/align-transcript":
            self._align_transcript()
        elif path == "/cleanup-transcript":
            self._cleanup_transcript()
        elif path == "/render":
            self._start_render()
        elif path == "/save-captions":
            self._save_captions()
        else:
            self._json(404, {"error": "Not found"})

    def _extract_audio(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self._json(413, {"error": "The video is empty or exceeds the 2 GB upload limit."})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._json(400, {"error": "Expected a multipart video upload."})
            return
        input_path = output_path = None
        job_id = None
        extraction_succeeded = False
        try:
            raw = self.rfile.read(content_length)
            message = BytesParser(policy=default).parsebytes((f"Content-Type: {content_type}\r\n\r\n").encode() + raw)
            upload = next((part for part in message.iter_parts() if part.get_param("name", header="content-disposition") == "video"), None)
            job_part = next((part for part in message.iter_parts() if part.get_param("name", header="content-disposition") == "job_id"), None)
            job_id = (job_part.get_content() or "").strip() if job_part else None
            if not self._valid_job_id(job_id):
                raise ValueError("A valid job_id is required.")
            if upload is None:
                raise ValueError("No video file was provided.")
            if not (upload.get_content_type() or "").startswith("video/"):
                raise ValueError("The uploaded file must be a video.")
            original_name = upload.get_filename() or "source-video"
            input_path = MEDIA_DIR / f"{uuid.uuid4().hex}{Path(original_name).suffix or '.video'}"
            output_name = f"{uuid.uuid4().hex}.mp3"
            output_path = MEDIA_DIR / output_name
            source_bytes = upload.get_payload(decode=True) or b""
            input_path.write_bytes(source_bytes)
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "input.mp4").write_bytes(source_bytes)
            completed = subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-codec:a", "libmp3lame", "-q:a", "2", str(output_path)], capture_output=True, text=True, timeout=600)
            if completed.returncode != 0:
                raise RuntimeError("ffmpeg could not extract audio from this video.")
            extraction_succeeded = True
            self._json(200, {"audio_url": f"/media/{output_name}", "filename": f"{Path(original_name).stem}.mp3"})
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            self._json(422, {"error": str(error)})
        finally:
            if input_path:
                input_path.unlink(missing_ok=True)
            if output_path and not extraction_succeeded:
                output_path.unlink(missing_ok=True)

    def _save_captions(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 10 * 1024 * 1024:
                raise ValueError("Invalid captions request.")
            payload = json.loads(self.rfile.read(content_length))
            job_id = payload.get("job_id")
            captions = payload.get("captions")
            if not self._valid_job_id(job_id) or not isinstance(captions, list) or not captions:
                raise ValueError("A valid job_id and caption pages are required.")
            job_dir = JOBS_DIR / job_id
            if not (job_dir / "input.mp4").is_file():
                raise ValueError("The source video for this job was not found.")
            (job_dir / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2))
            self._json(200, {"status": "saved", "job_id": job_id})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(422, {"error": str(error)})

    def _transcribe_audio(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64 * 1024:
                raise ValueError("Invalid transcription request.")
            payload = json.loads(self.rfile.read(content_length))
            requested = Path(urlparse(payload.get("audio_url", "")).path.lstrip("/"))
            audio_path = (ROOT / requested).resolve()
            if not requested.parts or requested.parts[0] != "media" or MEDIA_DIR not in audio_path.parents or not audio_path.is_file():
                raise ValueError("The extracted audio file was not found.")
            api_key = os.environ.get("SARVAM_API_KEY")
            if not api_key:
                raise RuntimeError("SARVAM_API_KEY is not configured on the server.")
            try:
                from sarvamai import SarvamAI
            except ImportError as error:
                raise RuntimeError("Install the Sarvam SDK with: pip install sarvamai") from error
            with tempfile.TemporaryDirectory(prefix="sarvam-") as work_dir:
                work_path = Path(work_dir)
                completed = subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-f", "segment", "-segment_time", str(TRANSCRIPTION_CHUNK_SECONDS), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(work_path / "chunk-%04d.wav")], capture_output=True, text=True, timeout=600)
                if completed.returncode != 0:
                    raise RuntimeError("ffmpeg could not prepare the audio for transcription.")
                client = SarvamAI(api_subscription_key=api_key)
                transcripts = []
                chunks = sorted(work_path.glob("chunk-*.wav"))
                if not chunks:
                    raise RuntimeError("No audio was available to transcribe.")
                for chunk in chunks:
                    with chunk.open("rb") as audio_file:
                        response = client.speech_to_text.transcribe(file=audio_file, model="saaras:v4", language_code="hi-IN", mode="transcribe")
                    text = getattr(response, "transcript", "") or ""
                    if text.strip():
                        transcripts.append(text.strip())
            self._json(200, {"transcript": " ".join(transcripts), "language_code": "hi-IN", "mode": "transcribe"})
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            self._json(422, {"error": str(error)})

    def _align_with_ctc(self, audio_path, transcript):
        try:
            import torch
            from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                             load_alignment_model, load_audio, postprocess_results,
                                             preprocess_text)
        except ImportError as error:
            raise RuntimeError("CTC forced aligner is not installed. Run: python -m pip install -r requirements.txt") from error
        device = os.environ.get("CTC_ALIGNER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        model, tokenizer = load_alignment_model(device, dtype=dtype)
        waveform = load_audio(str(audio_path), model.dtype, model.device)
        emissions, stride = generate_emissions(model, waveform, batch_size=16)
        original_words = transcript.split()
        tokens_starred, text_starred = preprocess_text(transcript, romanize=True, language="hin")
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        aligned = postprocess_results(text_starred, spans, stride, scores)
        if len(aligned) != len(original_words):
            raise RuntimeError(f"CTC aligner returned {len(aligned)} words for {len(original_words)} transcript words.")
        return [{"word": word, "start": item["start"], "end": item["end"]} for word, item in zip(original_words, aligned)]

    def _align_transcript(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 2 * 1024 * 1024:
                raise ValueError("Invalid alignment request.")
            payload = json.loads(self.rfile.read(content_length))
            transcript = payload.get("transcript")
            aligner = payload.get("aligner", "whisperx")
            if aligner not in ALIGNMENT_OPTIONS:
                raise ValueError("Invalid alignment engine.")
            requested = Path(urlparse(payload.get("audio_url", "")).path.lstrip("/"))
            audio_path = (ROOT / requested).resolve()
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("A transcript is required before alignment.")
            if not requested.parts or requested.parts[0] != "media" or MEDIA_DIR not in audio_path.parents or not audio_path.is_file():
                raise ValueError("The extracted audio file was not found.")
            if aligner == "ctc":
                timed_words = self._align_with_ctc(audio_path, transcript)
                self._json(200, {"words": timed_words, "word_count": len(timed_words), "language": "hi", "aligner": aligner})
                return
            try:
                import whisperx
            except ImportError as error:
                raise RuntimeError("Install WhisperX with: pip install whisperx") from error
            global _whisperx_align_model, _whisperx_metadata
            if _whisperx_align_model is None:
                _whisperx_align_model, _whisperx_metadata = whisperx.load_align_model(language_code="hi", device=WHISPERX_DEVICE)
            audio = whisperx.load_audio(str(audio_path))
            transcript_words = transcript.split()
            if not transcript_words:
                raise ValueError("The transcript has no words to align.")
            segment_words = []
            current = []
            for word in transcript_words:
                current.append(word)
                if len(current) >= 12 or re.search(r"[.!?।]$", word):
                    segment_words.append(current)
                    current = []
            if current:
                segment_words.append(current)
            duration = len(audio) / 16000.0
            segments = []
            cursor = 0.0
            for index, words in enumerate(segment_words):
                end = duration if index == len(segment_words) - 1 else cursor + duration * len(words) / len(transcript_words)
                segments.append({"start": cursor, "end": end, "text": " ".join(words)})
                cursor = end
            aligned = whisperx.align(segments, _whisperx_align_model, _whisperx_metadata, audio, WHISPERX_DEVICE, return_char_alignments=False)
            words = aligned.get("word_segments", [])
            if not words:
                for segment in aligned.get("segments", []):
                    words.extend(segment.get("words", []))
            timed_words = [{"word": word.get("word", "").strip(), "start": word["start"], "end": word["end"]} for word in words if word.get("word", "").strip() and "start" in word and "end" in word]
            if not timed_words:
                raise RuntimeError("WhisperX did not return word-level timings.")
            self._json(200, {"words": timed_words, "word_count": len(timed_words), "language": "hi", "aligner": aligner})
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            self._json(422, {"error": str(error)})

    def _cleanup_transcript(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 2 * 1024 * 1024:
                raise ValueError("Invalid cleanup request.")
            transcript = json.loads(self.rfile.read(content_length)).get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("A transcript is required before cleanup.")
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured on the server.")
            words = transcript.split()
            max_calls = 5
            chunk_size = max(CLEANUP_WORDS_PER_REQUEST, (len(words) + max_calls - 1) // max_calls)
            cleaned_words = []
            for start in range(0, len(words), chunk_size):
                chunk = words[start:start + chunk_size]
                numbered = " ".join(f"{start + index + 1}:{word}" for index, word in enumerate(chunk))
                expected = [str(start + index + 1) for index in range(len(chunk))]
                schema = {"type": "OBJECT", "properties": {key: {"type": "STRING"} for key in expected}, "required": expected}
                prompt = CLEANUP_SYSTEM_PROMPT + "\n\nINPUT:\n" + numbered
                request_body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 2048, "responseMimeType": "application/json", "responseSchema": schema}}, ensure_ascii=False).encode("utf-8")
                request = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent", data=request_body, method="POST", headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        provider_response = json.loads(response.read())
                except urllib.error.HTTPError as error:
                    if error.code == 429:
                        raise RuntimeError("Gemini rate limit exceeded. Wait for the provider quota to reset before retrying.") from error
                    raise RuntimeError(f"Gemini request failed with HTTP {error.code}.") from error
                candidate = provider_response.get("candidates", [{}])[0]
                print(f"[Gemini REST finish reason chunk {start + 1}-{start + len(chunk)}] {candidate.get('finishReason', 'unknown')}", flush=True)
                parts = candidate.get("content", {}).get("parts", [])
                content = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
                print(f"[Gemini REST raw chunk {start + 1}-{start + len(chunk)}] {content!r}", flush=True)
                if not content:
                    raise RuntimeError("Gemini returned no text content.")
                fence = chr(96) * 3
                if content.startswith(fence):
                    content = content.split(chr(10), 1)[1].rsplit(fence, 1)[0].strip()
                try:
                    mapped = json.loads(content)
                except (IndexError, json.JSONDecodeError) as error:
                    raise RuntimeError("Gemini returned invalid JSON for a cleanup chunk.") from error
                if set(mapped) != set(expected) or not all(isinstance(mapped[key], str) for key in expected):
                    raise RuntimeError("Gemini returned the wrong number of numbered words.")
                cleaned_words.extend(mapped[key] for key in expected)
            self._json(200, {"model": GEMINI_MODEL, "cleaned_transcript": " ".join(cleaned_words), "word_count": len(cleaned_words)})
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            status = 429 if "rate limit" in str(error).lower() else 422
            self._json(status, {"error": str(error)}, {"Retry-After": "31"} if status == 429 else None)

    def _render_status_path(self, path):
        parts = Path(path.lstrip("/")).parts
        if len(parts) != 3 or parts[0] != "render" or parts[2] != "status":
            return None
        job_id = parts[1]
        if not self._valid_job_id(job_id):
            self._json(400, {"error": "Invalid job_id."})
            return False
        status_path = JOBS_DIR / job_id / "status.json"
        if not status_path.is_file():
            self._json(404, {"error": "Render job not found."})
            return False
        return json.loads(status_path.read_text())

    @staticmethod
    def _valid_job_id(job_id):
        return isinstance(job_id, str) and bool(job_id) and all(char.isalnum() or char in "-_" for char in job_id)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_status(job_dir, payload):
        status_path = job_dir / "status.json"
        temp_path = job_dir / "status.json.tmp"
        temp_path.write_text(json.dumps(payload, ensure_ascii=False))
        temp_path.replace(status_path)

    def _start_render(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64 * 1024:
                raise ValueError("Invalid render request.")
            payload = json.loads(self.rfile.read(content_length))
            job_id = payload.get("job_id")
            if not self._valid_job_id(job_id):
                self._json(400, {"error": "Invalid job_id."})
                return
            job_dir = JOBS_DIR / job_id
            captions_path = job_dir / "captions.json"
            input_path = job_dir / "input.mp4"
            if not captions_path.is_file() or not input_path.is_file():
                self._json(404, {"error": "Job captions.json and input.mp4 are required."})
                return
            try:
                style = build_style(payload.get("font"), payload.get("accent"), payload.get("position"), payload.get("font_size", "standard"), payload.get("animation", "word_highlight"))
            except KeyError as error:
                self._json(400, {"error": str(error)})
                return
            with RENDER_LOCK:
                status_path = job_dir / "status.json"
                if status_path.is_file():
                    existing = json.loads(status_path.read_text())
                    if existing.get("state") in {"queued", "rendering"}:
                        self._json(409, {"error": "A render for this job is already in progress.", "job_id": job_id})
                        return
                (job_dir / "style.json").write_text(json.dumps(style, indent=2))
                self._write_status(job_dir, {"state": "queued", "queued_at": self._now()})
                RENDER_EXECUTOR.submit(self._run_render, job_id)
            self._json(202, {"status": "queued", "job_id": job_id})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._json(422, {"error": str(error)})

    def _run_render(self, job_id):
        job_dir = JOBS_DIR / job_id
        output_path = job_dir / "output.mp4"
        log_path = job_dir / "render.log"
        output_path.unlink(missing_ok=True)
        self._write_status(job_dir, {"state": "rendering", "started_at": self._now()})
        composition = os.environ.get("HYPERFRAMES_COMPOSITION", str(ROOT / "caption_composition.html"))
        composition_path = Path(composition).expanduser().resolve()
        if not composition_path.is_file():
            self._write_status(job_dir, {"state": "failed", "error": f"HyperFrames composition not found: {composition_path}"})
            return
        variables_path = job_dir / "render-variables.json"
        variables_path.write_text(json.dumps({
            "input_video": (job_dir / "input.mp4").relative_to(ROOT).as_posix(),
            "captions": (job_dir / "captions.json").read_text(),
            "style": (job_dir / "style.json").read_text(),
        }, ensure_ascii=False))
        command = ["npx", "--yes", "hyperframes", "render", str(composition_path.parent), "--composition", composition_path.name, "--variables-file", str(variables_path), "--strict-variables", "--output", str(output_path)]
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(command, cwd=str(ROOT), env={**os.environ, "CI": "1", "NPM_CONFIG_YES": "true"}, stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=3600)
            if result.returncode == 0 and output_path.is_file():
                self._write_status(job_dir, {"state": "done", "output": "output.mp4", "finished_at": self._now()})
            else:
                lines = log_path.read_text(errors="replace").splitlines()[-40:]
                self._write_status(job_dir, {"state": "failed", "error": chr(10).join(lines) or "HyperFrames render failed."})
        except FileNotFoundError:
            self._write_status(job_dir, {"state": "failed", "error": "npx was not found. Install Node.js and HyperFrames."})
        except subprocess.TimeoutExpired:
            self._write_status(job_dir, {"state": "failed", "error": "HyperFrames render timed out after 3600 seconds."})
        except OSError as error:
            self._write_status(job_dir, {"state": "failed", "error": str(error)})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/render/options":
            self._json(200, public_options())
            return
        if path == "/align/options":
            self._json(200, {"aligners": ALIGNMENT_OPTIONS})
            return
        render_status = self._render_status_path(path)
        if render_status is not None and render_status is not False:
            self._json(200, render_status)
            return
        requested = Path(path.lstrip("/"))
        if requested.parts and requested.parts[0] == "media":
            file_path = (ROOT / requested).resolve()
            if MEDIA_DIR not in file_path.parents or not file_path.is_file():
                self._json(404, {"error": "Audio file not found."})
                return
        super().do_GET()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "4173"))
    print(f"Workflow server running at http://localhost:{port}")
    print("Requires ffmpeg, SARVAM_API_KEY, and GEMINI_API_KEY for all workflow steps.")
    ThreadingHTTPServer(("127.0.0.1", port), WorkflowHandler).serve_forever()
