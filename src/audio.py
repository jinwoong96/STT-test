"""마이크 캡처 + 무음 기반 문장 분할(VAD).

오디오 콜백은 절대 무거운 일을 하지 않는다. 프레임을 큐에 넣기만 하고,
말/무음 판정과 조각 자르기는 별도 워커 스레드에서 처리한다.
"""
from __future__ import annotations

import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

from .config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Settings

# 무음 판정의 하한선. 이보다 조용하면 주변 소음이 아무리 낮아도 말로 치지 않는다.
ABSOLUTE_FLOOR_DB = -55.0
PREROLL_SEC = 0.4          # 말이 시작되기 직전 구간도 붙여줘야 첫 음절이 안 잘린다
SPEECH_ONSET_FRAMES = 3    # 연속 3프레임(90ms) 이상이어야 말 시작으로 인정


@dataclass
class AudioDevice:
    index: int
    name: str
    channels: int

    def __str__(self) -> str:  # 콤보박스에 그대로 표시된다
        return f"[{self.index}] {self.name}"


def list_input_devices() -> list[AudioDevice]:
    """입력 채널이 있는 장치만 추린다."""
    devices: list[AudioDevice] = []
    try:
        for index, info in enumerate(sd.query_devices()):
            channels = int(info.get("max_input_channels", 0))
            if channels > 0:
                devices.append(AudioDevice(index, str(info["name"]), channels))
    except Exception:
        return []
    return devices


def default_input_device() -> AudioDevice | None:
    try:
        default_index = sd.default.device[0]
    except Exception:
        default_index = None
    devices = list_input_devices()
    if not devices:
        return None
    for device in devices:
        if device.index == default_index:
            return device
    return devices[0]


def _dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))
    return 20.0 * math.log10(max(rms, 1e-7))


class MicListener:
    """마이크를 열고, 말이 끝날 때마다 발화 조각을 콜백으로 넘긴다."""

    def __init__(
        self,
        settings: Settings,
        on_segment: Callable[[np.ndarray, float, int], None],
        on_level: Callable[[float, bool], None],
        on_error: Callable[[str], None],
        on_partial: Callable[[np.ndarray, int], None] | None = None,
    ) -> None:
        self._settings = settings
        self._on_segment = on_segment
        self._on_level = on_level
        self._on_error = on_error
        # 말하는 도중 지금까지의 발화를 통째로 넘겨주는 콜백(미리보기용).
        # 발화마다 번호를 붙여, 최종 결과보다 늦게 도착한 미리보기를 버릴 수 있게 한다.
        self._on_partial = on_partial
        self._utterance_id = 0

        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None
        self._noise_floor_db = ABSOLUTE_FLOOR_DB

    # ---------- 수명주기 ----------
    def start(self, device_index: int | None) -> None:
        if self._worker is not None:
            return
        self._stop.clear()
        self._noise_floor_db = ABSOLUTE_FLOOR_DB
        while not self._frames.empty():           # 이전 세션 잔여 프레임 제거
            self._frames.get_nowait()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=device_index,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._worker = threading.Thread(target=self._segment_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._frames.put(None)                    # 워커 깨우기
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    @property
    def running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ---------- 내부 ----------
    def _audio_callback(self, indata, _frames, _time, status) -> None:
        if status and status.input_overflow:
            pass  # 한두 프레임 흘리는 건 치명적이지 않다
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _segment_loop(self) -> None:
        settings = self._settings
        preroll_len = max(1, int(PREROLL_SEC * 1000 / FRAME_MS))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_len)

        buffer: list[np.ndarray] = []
        speaking = False
        onset_run = 0
        silence_run = 0
        speech_frames = 0      # 조각 안에서 실제로 '말'로 판정된 프레임 수
        partial_at = 0         # 다음 미리보기를 낼 버퍼 길이(프레임 수)
        partial_step = max(1, int(settings.interim_interval_sec * 1000 / FRAME_MS))

        try:
            while not self._stop.is_set():
                frame = self._frames.get()
                if frame is None:
                    break

                level_db = _dbfs(frame)
                threshold = max(
                    self._noise_floor_db + settings.sensitivity_db, ABSOLUTE_FLOOR_DB
                )
                is_speech = level_db > threshold

                # 조용한 프레임으로만 소음 바닥값을 천천히 갱신한다
                if not is_speech:
                    self._noise_floor_db = 0.95 * self._noise_floor_db + 0.05 * level_db
                    self._noise_floor_db = max(self._noise_floor_db, -80.0)

                self._on_level(level_db, is_speech and speaking)

                if not speaking:
                    preroll.append(frame)
                    onset_run = onset_run + 1 if is_speech else 0
                    if onset_run >= SPEECH_ONSET_FRAMES:
                        speaking = True
                        silence_run = 0
                        buffer = list(preroll)     # 앞부분 붙여서 시작
                        speech_frames = onset_run
                        partial_at = partial_step
                        preroll.clear()
                    continue

                buffer.append(frame)
                if is_speech:
                    speech_frames += 1
                    silence_run = 0
                else:
                    silence_run += 1

                # 말하는 도중 미리보기: 지금까지 모인 발화 전체를 넘긴다.
                # 접두사 전체를 보내야 미리보기가 잘린 문장이 아니라 온전한 문장이 된다.
                if (
                    self._on_partial is not None
                    and settings.interim_enabled
                    and len(buffer) >= partial_at
                    and speech_frames * FRAME_MS / 1000 >= settings.min_segment_sec
                ):
                    partial_at = len(buffer) + partial_step
                    self._on_partial(
                        np.concatenate(buffer).astype(np.float32, copy=False),
                        self._utterance_id,
                    )

                silence_sec = silence_run * FRAME_MS / 1000
                buffered_sec = len(buffer) * FRAME_MS / 1000

                if silence_sec >= settings.silence_hold_sec:
                    self._flush(buffer, speech_frames, trim_tail=silence_run)
                    buffer, speaking = [], False
                    onset_run = silence_run = speech_frames = 0
                elif buffered_sec >= settings.max_segment_sec:
                    # 쉬지 않고 말하는 중 — 잘라 보내되 말하는 상태는 유지
                    self._flush(buffer, speech_frames, trim_tail=0)
                    buffer, silence_run, speech_frames = [], 0, 0
                    partial_at = partial_step

            if buffer:
                self._flush(buffer, speech_frames, trim_tail=0)
        except Exception as exc:  # 워커가 조용히 죽으면 원인을 알 수 없다
            self._on_error(f"오디오 처리 오류: {exc}")

    def _flush(self, buffer: list[np.ndarray], speech_frames: int, trim_tail: int) -> None:
        if not buffer:
            return
        # 길이 판정은 앞뒤 무음을 뺀 '실제 말한 시간' 으로 한다.
        # 버퍼 전체로 재면 짧은 클릭 잡음도 preroll+무음 때문에 길어 보인다.
        speech_sec = speech_frames * FRAME_MS / 1000
        if speech_sec < self._settings.min_segment_sec:
            return
        # 끝쪽 무음은 절반만 남긴다 (문장 끝 여운은 인식에 도움이 된다)
        keep = len(buffer) - trim_tail // 2
        chunk = buffer[:keep] if keep > 0 else buffer
        audio = np.concatenate(chunk).astype(np.float32, copy=False)
        self._on_segment(audio, len(audio) / SAMPLE_RATE, self._utterance_id)
        self._utterance_id += 1
