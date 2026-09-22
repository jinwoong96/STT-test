"""모델을 미리 내려받아 두는 스크립트.

앱에서도 첫 실행 시 자동으로 받지만, 미리 받아두면 처음 눌렀을 때 바로 시작된다.

    python download_model.py                # 기본 모델(large-v3-turbo)
    python download_model.py large-v3       # 특정 모델
    python download_model.py --all          # 전부
"""
from __future__ import annotations

import sys

from src.config import DEFAULT_MODEL, MODEL_CATALOG, MODELS_DIR, ensure_models_dir
from src.transcriber import model_is_downloaded


def download(model_name: str) -> None:
    from faster_whisper import WhisperModel

    if model_is_downloaded(model_name):
        print(f"[건너뜀] {model_name} — 이미 받아져 있습니다")
        return
    meta = MODEL_CATALOG.get(model_name, {})
    print(f"[다운로드] {model_name} ({meta.get('size_hint', '크기 미상')}) ...")
    WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS_DIR / "hf"),
    )
    print(f"[완료] {model_name}")


def main() -> int:
    ensure_models_dir()
    args = sys.argv[1:]
    if args and args[0] == "--all":
        targets = list(MODEL_CATALOG)
    elif args:
        targets = args
    else:
        targets = [DEFAULT_MODEL]

    for name in targets:
        if name not in MODEL_CATALOG:
            print(f"[오류] 알 수 없는 모델: {name}")
            print("사용 가능: " + ", ".join(MODEL_CATALOG))
            return 1
        download(name)

    print(f"\n저장 위치: {MODELS_DIR / 'hf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
