"""faster-whisper 래퍼 — 모델 로드와 발화 조각 → 텍스트 변환."""
from __future__ import annotations

import re
import threading
from typing import Callable

import numpy as np

from .config import MODEL_CATALOG, MODELS_DIR, Settings, ensure_models_dir

# Whisper 가 무음/잡음 구간에서 습관적으로 뱉는 문구들.
# 실시간 모드에서는 이런 조각이 계속 끼어들어 결과를 망친다.
_HALLUCINATIONS = (
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "구독 좋아요 부탁드립니다",
    "다음 영상에서 만나요",
    "한글자막 by",
    "자막 제공",
    "mbc 뉴스",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by",
    "amara.org",
)
# 의미 없는 반복(아아아아, ㅋㅋㅋㅋ, ......)
_REPEAT_RE = re.compile(r"^(.)\1{3,}$")


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if any(pattern in lowered for pattern in _HALLUCINATIONS):
        return True
    compact = re.sub(r"[\s.,!?·…]", "", stripped)
    if not compact:
        return True
    return bool(_REPEAT_RE.match(compact))


# download_root 를 주면 hub 하위 폴더 없이 여기에 바로 models--* 가 생긴다.
# HF_HOME 로 동작할 때를 대비해 hub 하위도 같이 훑는다.
_CACHE_ROOTS = ("hf", "hf/hub")


def model_is_downloaded(model_name: str) -> bool:
    """이 모델이 이미 로컬에 받아져 있는지."""
    ensure_models_dir()
    repo = MODEL_CATALOG.get(model_name, {}).get("repo")
    if not repo:
        return False
    # 부분 일치로 판별하면 large-v3 가 large-v3-turbo 폴더에 걸린다. 정확히 맞춘다.
    folder = "models--" + repo.replace("/", "--")
    for relative in _CACHE_ROOTS:
        entry = MODELS_DIR / relative / folder
        if entry.is_dir() and any((entry / "snapshots").glob("*/model.bin")):
            return True
    return False


class Transcriber:
    """모델 하나를 들고 있으면서 오디오 조각을 텍스트로 바꾼다.

    최종 변환용과 미리보기용으로 두 개를 따로 만들어 동시에 돌린다.
    실측(Ryzen 5 7600): num_workers=2 면 처리량 1.49배, 3이면 2.06배.
    small 을 large-v3-turbo 와 동시에 돌려도 turbo 는 0.24초만 느려진다.
    """

    def __init__(self, cpu_threads: int = 6, num_workers: int = 1) -> None:
        self._model = None
        self._model_name: str | None = None
        self._lock = threading.Lock()
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers

    @property
    def loaded_model(self) -> str | None:
        return self._model_name

    def load(self, model_name: str, on_status: Callable[[str], None]) -> None:
        """모델을 메모리에 올린다. 없으면 이때 자동으로 내려받는다."""
        with self._lock:
            if self._model_name == model_name and self._model is not None:
                return
            ensure_models_dir()
            from faster_whisper import WhisperModel  # 지연 임포트 — GUI 를 빨리 띄우려고

            if model_is_downloaded(model_name):
                on_status(f"모델 불러오는 중: {model_name}")
            else:
                on_status(f"모델 다운로드 중: {model_name} — 처음 한 번만 받습니다")

            self._model = None  # 교체 전 이전 모델 메모리 해제
            self._model_name = None
            self._model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",       # AMD GPU 환경이라 CPU int8 이 가장 현실적
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,   # 동시 변환 허용 (밀린 조각을 빨리 뺀다)
                download_root=str(MODELS_DIR / "hf"),
            )
            self._model_name = model_name
            on_status(f"준비 완료: {model_name}")

    def transcribe(self, audio: np.ndarray, settings: Settings, quick: bool = False) -> str:
        """quick=True 면 미리보기용 — 정확도를 조금 내주고 속도를 택한다."""
        with self._lock:
            model = self._model
        if model is None:
            return ""

        language = None if settings.language == "auto" else settings.language
        segments, _info = model.transcribe(
            audio,
            language=language,
            beam_size=1 if quick else settings.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            # 실시간에서는 이전 문맥을 물리면 한 번 튄 오인식이 계속 번진다
            condition_on_previous_text=False,
            initial_prompt=settings.initial_prompt or None,
            temperature=[0.0, 0.2, 0.4],
            no_speech_threshold=0.6,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        text = re.sub(r"\s{2,}", " ", text)
        if settings.filter_hallucination and is_noise_text(text):
            return ""
        return text

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._model_name = None
