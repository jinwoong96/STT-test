"""앱 전역 설정과 모델 카탈로그."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 프로젝트 루트 / 모델 저장 위치 (모델은 전부 이 폴더에 로컬로 받음)
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
SETTINGS_PATH = ROOT_DIR / "settings.json"

# 오디오 캡처 규격 — Whisper 계열은 16kHz 모노 입력을 전제로 한다
SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 샘플

# 선택 가능한 모델.
#   repo      HuggingFace 저장소 (로컬에 받았는지 정확히 판별하는 데 쓴다)
#   size_hint 디스크 사용량
#   latency   이 PC(Ryzen 5 7600 · CPU int8)에서 실측한 호출당 고정비용.
#             Whisper 인코더는 조각 길이와 무관하게 늘 30초 분량을 처리하므로,
#             말을 멈춘 뒤 글자가 뜨기까지 걸리는 시간이 사실상 이 값이다.
MODEL_CATALOG: dict[str, dict[str, str]] = {
    "large-v3-turbo": {
        "label": "large-v3-turbo — 정확도 높음 / 반응 4.3초 (권장)",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "latency": "4.3초",
        "size_hint": "약 1.6GB",
    },
    "large-v3": {
        "label": "large-v3 — 최고 정확도 / 반응 6초+ (느림)",
        "repo": "Systran/faster-whisper-large-v3",
        "latency": "6초 이상",
        "size_hint": "약 3.1GB",
    },
    "medium": {
        "label": "medium — 절충 / 반응 2.8초",
        "repo": "Systran/faster-whisper-medium",
        "latency": "2.8초",
        "size_hint": "약 1.5GB",
    },
    "small": {
        "label": "small — 가장 빠름 / 반응 1.1초 (정확도 낮음)",
        "repo": "Systran/faster-whisper-small",
        "latency": "1.1초",
        "size_hint": "약 0.5GB",
    },
}
DEFAULT_MODEL = "large-v3-turbo"

LANGUAGE_CHOICES: dict[str, str] = {
    "ko": "한국어",
    "en": "영어",
    "ja": "일본어",
    "zh": "중국어",
    "auto": "자동 감지",
}


@dataclass
class Settings:
    """사용자가 GUI에서 바꾸는 값들. settings.json 에 저장된다."""

    model_name: str = DEFAULT_MODEL
    language: str = "ko"
    device_index: int | None = None
    device_name: str = ""

    # VAD(말/무음 구분) 튜닝값
    sensitivity_db: float = 10.0      # 주변 소음 대비 몇 dB 위를 '말'로 볼지
    silence_hold_sec: float = 0.8     # 이만큼 조용하면 한 문장이 끝난 것으로 본다
    max_segment_sec: float = 20.0     # 한 조각의 최대 길이 (계속 말할 때 강제로 끊음)
    min_segment_sec: float = 0.35     # 이보다 짧으면 잡음으로 보고 버림

    # 말하는 도중 미리보기(중간 결과)
    # 작은 모델을 병렬로 돌려 회색 글씨로 먼저 띄우고, 말이 끝나면 본 모델이 교체한다.
    # 실측상 small 을 large-v3-turbo 와 동시에 돌려도 최종 변환은 0.24초만 느려진다.
    interim_enabled: bool = True
    interim_model: str = "small"
    interim_interval_sec: float = 2.0   # 말하는 중 이 간격마다 미리보기 갱신

    # 인식 품질
    beam_size: int = 5
    initial_prompt: str = ""
    filter_hallucination: bool = True

    # 화면
    autoscroll: bool = True
    show_timestamp: bool = True

    _unknown: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            return cls()
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        try:
            SETTINGS_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # 설정 저장 실패가 앱을 막을 이유는 없다


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # huggingface_hub 캐시도 프로젝트 안으로 묶어 완전한 로컬 동작을 보장
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    return MODELS_DIR
