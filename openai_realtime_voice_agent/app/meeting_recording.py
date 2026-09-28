"""Privacy-first meeting recording and diarized transcription."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)


def get_meeting_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "start_meeting_recording",
            "description": (
                "Start a meeting recording. Before calling, ALWAYS ask how many people are present and "
                "whether speaker names are wanted. If names are wanted, collect every name first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "participant_count": {"type": "integer", "minimum": 1, "maximum": 50},
                    "identify_by_name": {"type": "boolean"},
                    "participant_names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["participant_count", "identify_by_name", "participant_names"],
            },
        },
        {
            "type": "function",
            "name": "stop_meeting_recording",
            "description": "Stop the active meeting recording and start diarized transcription.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]


class MeetingRecordingService:
    """Owns one local PCM recording and the post-stop transcription job."""

    def __init__(self, api_key: str, base_dir: str = "/share/voice-meetings"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active = False
        self.command_window = False
        self._file = None
        self._pcm_bytes = 0
        self._wav_path: Path | None = None
        self._meta: Dict[str, Any] = {}
        self._send_control: Callable[[dict], Awaitable[None]] | None = None

    def set_control_sender(self, sender: Callable[[dict], Awaitable[None]]) -> None:
        self._send_control = sender

    async def start(self, participant_count: int, identify: bool, names: List[str]) -> str:
        if self.active:
            return "Er loopt al een opname."
        count = max(1, min(50, int(participant_count)))
        clean_names = [str(n).strip() for n in names if str(n).strip()]
        if identify and len(clean_names) != count:
            return f"Ik heb {count} aanwezigen maar {len(clean_names)} namen. Vraag eerst alle namen."

        now = datetime.now()
        meeting_dir = self.base_dir / now.strftime("%Y-%m-%d")
        meeting_dir.mkdir(parents=True, exist_ok=True)
        stem = now.strftime("meeting-%H%M%S")
        self._wav_path = meeting_dir / f"{stem}.wav"
        self._file = self._wav_path.open("wb")
        self._write_wav_header(self._file, 16000)
        self._pcm_bytes = 0
        self._meta = {
            "started_at": now.isoformat(),
            "participant_count": count,
            "identify_by_name": bool(identify),
            "participant_names": clean_names,
        }
        self.active = True
        self.command_window = False
        if self._send_control:
            await self._send_control({"type": "meeting", "value": "start"})
        logger.info("⏺️ Meeting recording started: %s", self._wav_path)
        if identify:
            joined = ", ".join(clean_names)
            return (
                "De opname loopt. Laat nu iedereen één voor één zeggen: mijn naam is, gevolgd door de naam. "
                f"Ik verwacht: {joined}."
            )
        return "De opname loopt. Sprekers worden aangeduid als Spreker 1, Spreker 2, enzovoort."

    def on_wake(self) -> None:
        if self.active:
            self.command_window = True
            logger.info("🎛️ Meeting command window opened by local wake word")

    def consume_audio(self, pcm: bytes) -> bool:
        """Record audio; return True only when it should also reach Realtime."""
        if not self.active:
            return True
        if self._file and pcm:
            self._file.write(pcm)
            self._pcm_bytes += len(pcm)
        return self.command_window

    async def stop(self, reason: str = "voice") -> str:
        if not self.active:
            return "Er loopt geen opname."
        self.active = False
        self.command_window = False
        if self._send_control:
            await self._send_control({"type": "meeting", "value": "stop"})
        self._finalize_wav()
        wav_path = self._wav_path
        logger.info("⏹️ Meeting recording stopped (%s): %s", reason, wav_path)
        asyncio.create_task(self._transcribe_and_store(wav_path, dict(self._meta)))
        return "De opname is gestopt. Ik maak nu het volledige transcript; dit kan enkele minuten duren."

    def finalize_for_shutdown(self) -> None:
        """Make an interrupted recording playable without deleting it."""
        if not self.active:
            return
        self.active = False
        self.command_window = False
        self._finalize_wav()
        logger.warning("⚠️ Add-on stopped during meeting; finalized source audio for recovery: %s", self._wav_path)

    async def _transcribe_and_store(self, wav_path: Path | None, meta: Dict[str, Any]) -> None:
        if not wav_path or not wav_path.exists():
            logger.error("Meeting WAV missing; cannot transcribe")
            return
        compressed = wav_path.with_suffix(".mp3")
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path),
                "-ac", "1", "-ar", "16000", "-b:a", "32k", str(compressed),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode(errors="replace"))
            audio_parts = [compressed]
            if compressed.stat().st_size >= 24 * 1024 * 1024:
                # The transcription endpoint has a per-upload ceiling. Segment
                # very long meetings instead of losing the entire transcript.
                pattern = compressed.with_name(compressed.stem + "-part-%03d.mp3")
                split = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(compressed),
                    "-f", "segment", "-segment_time", "4500", "-c", "copy", str(pattern),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, split_stderr = await split.communicate()
                if split.returncode != 0:
                    raise RuntimeError(split_stderr.decode(errors="replace"))
                audio_parts = sorted(compressed.parent.glob(compressed.stem + "-part-*.mp3"))
                if not audio_parts:
                    raise RuntimeError("Lange opname kon niet worden gesplitst")

            segments: List[dict] = []
            payload: Dict[str, Any] = {"chunks": []}
            offset = 0.0
            for part in audio_parts:
                with part.open("rb") as audio:
                    result = await self.client.audio.transcriptions.create(
                        model="gpt-4o-transcribe-diarize",
                        file=audio,
                        response_format="diarized_json",
                        chunking_strategy="auto",
                        language="nl",
                    )
                chunk = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                chunk_segments = chunk.get("segments") or []
                for seg in chunk_segments:
                    item = dict(seg)
                    item["start"] = float(item.get("start", 0)) + offset
                    item["end"] = float(item.get("end", item["start"])) + offset
                    # Avoid pretending labels from separate API jobs are the
                    # same voice; preserve the chunk in the stable raw label.
                    item["speaker"] = f"chunk{len(payload['chunks'])}_{item.get('speaker', 'unknown')}"
                    segments.append(item)
                duration = float(chunk.get("duration") or max(
                    [float(s.get("end", 0)) for s in chunk_segments] or [0]
                ))
                offset += duration
                payload["chunks"].append(chunk)
            if not segments:
                raise RuntimeError("Transcript bevat geen sprekersegmenten")
            name_map = self._infer_name_map(segments, meta.get("participant_names", []))
            lines = ["# Vergadertranscript", "", f"Start: {meta['started_at']}", ""]
            unknown_map: Dict[str, str] = {}
            for seg in segments:
                raw_speaker = str(seg.get("speaker", "unknown"))
                if raw_speaker not in name_map and raw_speaker not in unknown_map:
                    unknown_map[raw_speaker] = f"Spreker {len(name_map) + len(unknown_map) + 1}"
                speaker = name_map.get(raw_speaker) or unknown_map[raw_speaker]
                start = float(seg.get("start", 0))
                text = str(seg.get("text", "")).strip()
                if text:
                    lines.append(f"**[{self._timestamp(start)}] {speaker}:** {text}")
                    lines.append("")
            transcript = "\n".join(lines).strip() + "\n"
            if len(transcript) < 40:
                raise RuntimeError("Transcriptvalidatie mislukt: resultaat is leeg")

            json_path = wav_path.with_suffix(".json")
            md_path = wav_path.with_suffix(".md")
            tmp_json = json_path.with_suffix(".json.tmp")
            tmp_md = md_path.with_suffix(".md.tmp")
            payload["meeting"] = meta
            payload["speaker_name_map"] = name_map
            tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_md.write_text(transcript, encoding="utf-8")
            tmp_json.replace(json_path)
            tmp_md.replace(md_path)
            # User explicitly authorised deletion, but only after both durable outputs exist.
            if json_path.stat().st_size > 20 and md_path.stat().st_size > 20:
                wav_path.unlink(missing_ok=True)
                compressed.unlink(missing_ok=True)
                for part in audio_parts:
                    if part != compressed:
                        part.unlink(missing_ok=True)
            logger.info("✅ Meeting transcript saved: %s", md_path)
        except Exception as exc:
            logger.exception("❌ Meeting transcription failed; source audio retained: %s", exc)
            compressed.unlink(missing_ok=True)
            for part in compressed.parent.glob(compressed.stem + "-part-*.mp3"):
                part.unlink(missing_ok=True)

    @staticmethod
    def _infer_name_map(segments: List[dict], names: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for seg in segments:
            text = str(seg.get("text", ""))
            speaker = str(seg.get("speaker", "unknown"))
            normalized = re.sub(r"[^a-zà-ÿ0-9 ]", " ", text.lower())
            for name in names:
                if re.search(rf"\b{re.escape(name.lower())}\b", normalized) and speaker not in mapping:
                    mapping[speaker] = name
        return mapping

    @staticmethod
    def _timestamp(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

    def _finalize_wav(self) -> None:
        if not self._file:
            return
        self._file.flush()
        self._file.seek(4)
        self._file.write(struct.pack("<I", 36 + self._pcm_bytes))
        self._file.seek(40)
        self._file.write(struct.pack("<I", self._pcm_bytes))
        self._file.close()
        self._file = None

    @staticmethod
    def _write_wav_header(file, sample_rate: int) -> None:
        file.write(b"RIFF" + struct.pack("<I", 0) + b"WAVEfmt ")
        file.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        file.write(b"data" + struct.pack("<I", 0))


def create_meeting_tool_handlers(service: MeetingRecordingService):
    async def start_handler(params: "FunctionCallParams") -> None:
        args = params.arguments or {}
        result = await service.start(
            args.get("participant_count", 1),
            bool(args.get("identify_by_name", False)),
            args.get("participant_names") or [],
        )
        await params.result_callback(result)

    async def stop_handler(params: "FunctionCallParams") -> None:
        await params.result_callback(await service.stop("voice command"))

    return start_handler, stop_handler
