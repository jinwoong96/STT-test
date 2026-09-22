"""Tkinter GUI — 실시간 받아쓰기 창."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .audio import MicListener, default_input_device, list_input_devices
from .config import LANGUAGE_CHOICES, MODEL_CATALOG, ROOT_DIR, Settings
from .transcriber import Transcriber, model_is_downloaded

FONT_UI = ("Malgun Gothic", 10)
FONT_TEXT = ("Malgun Gothic", 12)
FONT_BIG = ("Malgun Gothic", 11, "bold")

METER_MIN_DB = -60.0
METER_MAX_DB = 0.0


class SttApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings.load()
        # 최종 변환용. num_workers=2 로 밀린 조각을 동시에 빼낸다.
        self.transcriber = Transcriber(cpu_threads=6, num_workers=2)
        # 말하는 도중 미리보기용 작은 모델. 스레드를 적게 줘서 최종 변환을 방해하지 않는다.
        self.interim_transcriber = Transcriber(cpu_threads=3, num_workers=1)

        self.ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.segment_queue: queue.Queue = queue.Queue()
        # 미리보기는 최신 것만 의미가 있다. 크기 1로 두고 밀리면 오래된 것을 버린다.
        self.partial_queue: queue.Queue = queue.Queue(maxsize=1)
        self.devices: list = []
        self.listener: MicListener | None = None
        self.worker: threading.Thread | None = None
        self.interim_worker: threading.Thread | None = None
        self.finalized_id = -1      # 여기까지는 확정됨 — 늦게 온 미리보기는 버린다
        self.listening = False
        self.busy = False              # 모델 로딩 등 오래 걸리는 작업 중인지
        self.pending = 0               # 변환 대기 중인 조각 수
        self._level_db = METER_MIN_DB
        self._level_active = False

        self.title("로컬 실시간 받아쓰기 (Whisper)")
        self.geometry("880x620")
        self.minsize(720, 480)
        self._build_ui()
        self._refresh_devices(select_saved=True)
        self._poll_ui_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=FONT_UI)
        style.configure("Big.TButton", font=FONT_BIG, padding=(18, 10))

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # --- 설정 줄 -------------------------------------------------
        setup = ttk.LabelFrame(outer, text="설정", padding=8)
        setup.pack(fill="x")
        setup.columnconfigure(1, weight=1)
        setup.columnconfigure(3, weight=1)

        ttk.Label(setup, text="모델").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(
            setup,
            textvariable=self.model_var,
            state="readonly",
            values=[meta["label"] for meta in MODEL_CATALOG.values()],
        )
        self.model_box.grid(row=0, column=1, columnspan=4, sticky="ew", pady=3)
        self.model_box.bind("<<ComboboxSelected>>", self._on_model_change)
        self._set_model_selection(self.settings.model_name)

        ttk.Label(setup, text="언어").grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.lang_var = tk.StringVar(
            value=LANGUAGE_CHOICES.get(self.settings.language, "한국어")
        )
        ttk.Combobox(
            setup,
            textvariable=self.lang_var,
            state="readonly",
            values=list(LANGUAGE_CHOICES.values()),
            width=12,
        ).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(setup, text="마이크").grid(row=1, column=2, sticky="e", padx=(12, 6))
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(
            setup, textvariable=self.device_var, state="readonly"
        )
        self.device_box.grid(row=1, column=3, sticky="ew", pady=3)
        ttk.Button(
            setup, text="새로고침", width=9, command=lambda: self._refresh_devices()
        ).grid(row=1, column=4, sticky="w", padx=(6, 0))

        # --- 민감도 --------------------------------------------------
        tune = ttk.Frame(setup)
        tune.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(6, 0))
        tune.columnconfigure(1, weight=1)
        tune.columnconfigure(4, weight=1)

        ttk.Label(tune, text="감지 민감도").grid(row=0, column=0, sticky="w")
        self.sens_var = tk.DoubleVar(value=self.settings.sensitivity_db)
        ttk.Scale(
            tune, from_=4.0, to=25.0, variable=self.sens_var,
            command=lambda _v: self._on_tune_change(),
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self.sens_label = ttk.Label(tune, width=8)
        self.sens_label.grid(row=0, column=2, sticky="w")

        ttk.Label(tune, text="문장 끊김 대기").grid(
            row=0, column=3, sticky="w", padx=(14, 0)
        )
        self.hold_var = tk.DoubleVar(value=self.settings.silence_hold_sec)
        ttk.Scale(
            tune, from_=0.3, to=2.5, variable=self.hold_var,
            command=lambda _v: self._on_tune_change(),
        ).grid(row=0, column=4, sticky="ew", padx=6)
        self.hold_label = ttk.Label(tune, width=8)
        self.hold_label.grid(row=0, column=5, sticky="w")
        self._on_tune_change()

        # --- 조작 줄 -------------------------------------------------
        control = ttk.Frame(outer)
        control.pack(fill="x", pady=(10, 6))

        self.toggle_btn = ttk.Button(
            control, text="▶  받아쓰기 시작", style="Big.TButton",
            command=self._toggle_listening,
        )
        self.toggle_btn.pack(side="left")

        meter_box = ttk.Frame(control)
        meter_box.pack(side="left", fill="x", expand=True, padx=14)
        ttk.Label(meter_box, text="입력 레벨").pack(anchor="w")
        self.meter = tk.Canvas(
            meter_box, height=16, bg="#e9e9ed",
            highlightthickness=1, highlightbackground="#c3c3c8",
        )
        self.meter.pack(fill="x")
        self._meter_bar = self.meter.create_rectangle(0, 0, 0, 20, fill="#4a90d9", width=0)

        # --- 결과 텍스트 ---------------------------------------------
        self.text = scrolledtext.ScrolledText(
            outer, wrap="word", font=FONT_TEXT, undo=True,
            padx=10, pady=8, relief="solid", borderwidth=1,
        )
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("ts", foreground="#8a8a93")
        # 말하는 도중 미리보기 — 확정되면 지워지고 제대로 된 문장으로 바뀐다
        self.text.tag_configure("interim", foreground="#9aa0a6")

        # --- 하단 ----------------------------------------------------
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(8, 0))

        ttk.Button(bottom, text="저장", command=self._save_text).pack(side="left")
        ttk.Button(bottom, text="전체 복사", command=self._copy_text).pack(
            side="left", padx=6
        )
        ttk.Button(bottom, text="지우기", command=self._clear_text).pack(side="left")

        self.ts_var = tk.BooleanVar(value=self.settings.show_timestamp)
        ttk.Checkbutton(bottom, text="시각 표시", variable=self.ts_var).pack(
            side="left", padx=(16, 0)
        )
        self.scroll_var = tk.BooleanVar(value=self.settings.autoscroll)
        ttk.Checkbutton(bottom, text="자동 스크롤", variable=self.scroll_var).pack(
            side="left", padx=8
        )
        self.interim_var = tk.BooleanVar(value=self.settings.interim_enabled)
        ttk.Checkbutton(
            bottom, text="말하는 중 미리보기", variable=self.interim_var,
            command=self._on_interim_toggle,
        ).pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="대기 중 — 시작을 누르면 모델을 불러옵니다")
        ttk.Label(
            outer, textvariable=self.status_var, foreground="#55555d", anchor="w"
        ).pack(fill="x", pady=(6, 0))

    # -------------------------------------------------------- 설정 헬퍼
    def _set_model_selection(self, model_name: str) -> None:
        meta = MODEL_CATALOG.get(model_name)
        if meta:
            self.model_var.set(meta["label"])

    def _selected_model(self) -> str:
        label = self.model_var.get()
        for name, meta in MODEL_CATALOG.items():
            if meta["label"] == label:
                return name
        return self.settings.model_name

    def _selected_language(self) -> str:
        label = self.lang_var.get()
        for code, name in LANGUAGE_CHOICES.items():
            if name == label:
                return code
        return "ko"

    def _on_model_change(self, _event=None) -> None:
        name = self._selected_model()
        meta = MODEL_CATALOG[name]
        if model_is_downloaded(name):
            note = "이미 받아져 있음"
        else:
            note = f"첫 실행 시 {meta['size_hint']} 다운로드"
        self._set_status(
            f"모델 선택: {name} · 말을 멈춘 뒤 약 {meta['latency']} 후 표시 · {note}"
        )

    def _on_interim_toggle(self) -> None:
        self.settings.interim_enabled = self.interim_var.get()
        if not self.settings.interim_enabled:
            self._clear_interim()

    def _on_tune_change(self) -> None:
        self.sens_label.config(text=f"{self.sens_var.get():.0f} dB")
        self.hold_label.config(text=f"{self.hold_var.get():.1f} 초")
        self.settings.sensitivity_db = float(self.sens_var.get())
        self.settings.silence_hold_sec = float(self.hold_var.get())

    def _refresh_devices(self, select_saved: bool = False) -> None:
        self.devices = list_input_devices()
        self.device_box["values"] = [str(d) for d in self.devices]
        if not self.devices:
            self.device_var.set("")
            self._set_status("입력 가능한 마이크를 찾지 못했습니다")
            return
        chosen = None
        if select_saved and self.settings.device_index is not None:
            chosen = next(
                (d for d in self.devices if d.index == self.settings.device_index), None
            )
        if chosen is None:
            chosen = default_input_device() or self.devices[0]
        self.device_var.set(str(chosen))

    def _selected_device_index(self) -> int | None:
        label = self.device_var.get()
        for device in self.devices:
            if str(device) == label:
                return device.index
        return None

    # ------------------------------------------------------- 시작 / 중지
    def _toggle_listening(self) -> None:
        if self.busy:
            return
        if self.listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self) -> None:
        if not self.devices:
            messagebox.showwarning("마이크 없음", "사용 가능한 마이크가 없습니다.")
            return

        self.settings.model_name = self._selected_model()
        self.settings.language = self._selected_language()
        self.settings.device_index = self._selected_device_index()
        self.settings.device_name = self.device_var.get()
        self.settings.show_timestamp = self.ts_var.get()
        self.settings.autoscroll = self.scroll_var.get()
        self.settings.interim_enabled = self.interim_var.get()
        self.settings.save()

        self.busy = True
        self.toggle_btn.config(text="준비 중…", state="disabled")
        self.model_box.config(state="disabled")
        self.device_box.config(state="disabled")

        threading.Thread(target=self._prepare_model, daemon=True).start()

    def _prepare_model(self) -> None:
        """모델 로드는 오래 걸리므로 워커 스레드에서. 완료를 UI 큐로 알린다."""
        try:
            self.transcriber.load(
                self.settings.model_name,
                lambda msg: self.ui_queue.put(("status", msg)),
            )
        except Exception as exc:
            self.ui_queue.put(("fatal", f"모델을 불러오지 못했습니다.\n\n{exc}"))
            return

        # 미리보기 모델은 있으면 좋고 없어도 되는 기능이다. 실패해도 본 기능은 계속한다.
        if self.settings.interim_enabled:
            interim_name = self.settings.interim_model
            if interim_name == self.settings.model_name:
                # 본 모델과 같으면 따로 띄울 이유가 없다
                self.ui_queue.put(("interim_off", "본 모델과 같아 미리보기 생략"))
            else:
                try:
                    self.interim_transcriber.load(
                        interim_name,
                        lambda msg: self.ui_queue.put(("status", f"미리보기 {msg}")),
                    )
                except Exception as exc:
                    self.ui_queue.put(
                        ("interim_off", f"미리보기 모델({interim_name}) 로드 실패: {exc}")
                    )
        self.ui_queue.put(("model_ready", None))

    def _begin_capture(self) -> None:
        """모델이 준비된 뒤 메인 스레드에서 호출된다."""
        self.finalized_id = -1
        self.worker = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.worker.start()
        self.interim_worker = threading.Thread(target=self._interim_loop, daemon=True)
        self.interim_worker.start()

        self.listener = MicListener(
            settings=self.settings,
            on_segment=lambda audio, dur, uid: self.segment_queue.put((audio, dur, uid)),
            on_level=self._on_level,
            on_error=lambda msg: self.ui_queue.put(("status", msg)),
            on_partial=self._on_partial,
        )
        try:
            self.listener.start(self.settings.device_index)
        except Exception as exc:
            self.listener = None
            self.segment_queue.put(None)
            self._reset_controls()
            messagebox.showerror("마이크 오류", f"마이크를 열지 못했습니다.\n\n{exc}")
            return

        self.listening = True
        self.busy = False
        self.toggle_btn.config(text="■  중지", state="normal")
        self._set_status(f"듣는 중 — {self.settings.model_name} · 말씀하세요")

    def _stop_listening(self) -> None:
        self.listening = False
        self.toggle_btn.config(text="정리 중…", state="disabled")
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self.segment_queue.put(None)       # 변환 워커에게 종료 신호
        self.partial_queue.put(None)       # 미리보기 워커에게도
        self._clear_interim()
        self._level_db, self._level_active = METER_MIN_DB, False
        self._draw_meter()
        self._reset_controls()
        self._set_status(
            f"중지됨 — 남은 {self.pending}개 변환 중…" if self.pending else "중지됨"
        )

    def _reset_controls(self) -> None:
        self.busy = False
        self.listening = False
        self.toggle_btn.config(text="▶  받아쓰기 시작", state="normal")
        self.model_box.config(state="readonly")
        self.device_box.config(state="readonly")

    # ------------------------------------------------------------ 변환
    def _transcribe_loop(self) -> None:
        while True:
            item = self.segment_queue.get()
            if item is None:
                break
            audio, duration, uid = item
            self.ui_queue.put(("pending", 1))
            started = time.perf_counter()
            try:
                text = self.transcriber.transcribe(audio, self.settings)
            except Exception as exc:
                self.ui_queue.put(("status", f"변환 오류: {exc}"))
                self.ui_queue.put(("pending", -1))
                continue
            elapsed = time.perf_counter() - started
            self.ui_queue.put(("pending", -1))
            # 텍스트가 비어도(잡음이었어도) 확정은 알려야 회색 미리보기가 걷힌다
            self.ui_queue.put(("text", (text, duration, elapsed, uid)))

    def _interim_loop(self) -> None:
        """말하는 도중 미리보기 전용 워커. 최종 변환과 병렬로 돈다."""
        while True:
            item = self.partial_queue.get()
            if item is None:
                break
            audio, uid = item
            if uid <= self.finalized_id or not self.settings.interim_enabled:
                continue          # 이미 확정된 발화 — 지금 돌려봐야 버려진다
            try:
                text = self.interim_transcriber.transcribe(
                    audio, self.settings, quick=True
                )
            except Exception:
                continue          # 미리보기 실패는 조용히 넘긴다
            if text:
                self.ui_queue.put(("interim", (text, uid)))

    def _on_partial(self, audio, uid: int) -> None:
        """세그먼터가 말하는 도중 호출한다. 최신 것만 남기고 밀린 건 버린다."""
        try:
            self.partial_queue.put_nowait((audio, uid))
        except queue.Full:
            try:
                self.partial_queue.get_nowait()
                self.partial_queue.put_nowait((audio, uid))
            except queue.Empty:
                pass

    def _on_level(self, level_db: float, active: bool) -> None:
        # 워커 스레드에서 33Hz 로 불린다. 값만 남기고 그리기는 UI 루프가 한다.
        self._level_db = level_db
        self._level_active = active

    # --------------------------------------------------------- UI 루프
    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "status":
                    self._set_status(str(payload))
                elif kind == "pending":
                    self.pending = max(0, self.pending + int(payload))
                    self._refresh_pending_status()
                elif kind == "text":
                    self._append_text(*payload)
                elif kind == "interim":
                    self._show_interim(*payload)
                elif kind == "interim_off":
                    self.settings.interim_enabled = False
                    self.interim_var.set(False)
                    self._set_status(str(payload))
                elif kind == "model_ready":
                    self._begin_capture()
                elif kind == "fatal":
                    self._reset_controls()
                    messagebox.showerror("오류", str(payload))
        except queue.Empty:
            pass
        self._draw_meter()
        self.after(60, self._poll_ui_queue)

    def _draw_meter(self) -> None:
        width = self.meter.winfo_width()
        if width <= 1:
            return
        ratio = (self._level_db - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)
        ratio = min(1.0, max(0.0, ratio))
        self.meter.coords(self._meter_bar, 0, 0, width * ratio, 20)
        self.meter.itemconfig(
            self._meter_bar, fill="#2fa84f" if self._level_active else "#4a90d9"
        )

    def _refresh_pending_status(self) -> None:
        if not self.listening:
            if self.pending == 0:
                self._set_status("중지됨")
            return
        base = f"듣는 중 — {self.settings.model_name}"
        self._set_status(f"{base} · 변환 대기 {self.pending}개" if self.pending else base)

    def _clear_interim(self) -> None:
        """회색 미리보기 글자를 걷어낸다."""
        span = self.text.tag_ranges("interim")
        if span:
            self.text.delete(span[0], span[-1])

    def _show_interim(self, text: str, uid: int) -> None:
        if uid <= self.finalized_id or not self.interim_var.get():
            return              # 이미 확정된 발화의 뒤늦은 미리보기 — 버린다
        self._clear_interim()
        self.text.insert("end", text, ("interim",))
        if self.scroll_var.get():
            self.text.see("end")

    def _append_text(self, text: str, duration: float, elapsed: float, uid: int) -> None:
        # 확정본이 왔으니 이 발화의 미리보기는 더 이상 의미가 없다
        self.finalized_id = max(self.finalized_id, uid)
        self._clear_interim()
        if not text:
            return              # 잡음이었던 조각 — 회색만 걷어내고 끝
        stamp = datetime.now().strftime("%H:%M:%S")
        if self.ts_var.get():
            self.text.insert("end", f"[{stamp}] ", ("ts",))
        self.text.insert("end", text + "\n")
        if self.scroll_var.get():
            self.text.see("end")
        speed = duration / elapsed if elapsed > 0 else 0.0
        state = "듣는 중" if self.listening else "중지됨"
        self._set_status(
            f"{state} — 방금 {duration:.1f}초를 {elapsed:.1f}초에 변환 (x{speed:.1f})"
        )

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    # ------------------------------------------------------------ 하단
    def _committed_text(self) -> str:
        """확정된 문장만. 아직 회색인 미리보기는 빼고 돌려준다.

        미리보기는 작은 모델이 대충 뽑은 임시 결과라 저장/복사에 섞이면 안 된다.
        """
        span = self.text.tag_ranges("interim")
        if not span:
            return self.text.get("1.0", "end").strip()
        return (self.text.get("1.0", span[0]) + self.text.get(span[-1], "end")).strip()

    def _save_text(self) -> None:
        content = self._committed_text()
        if not content:
            messagebox.showinfo("저장", "저장할 내용이 없습니다.")
            return
        default = f"받아쓰기_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            title="받아쓰기 저장", defaultextension=".txt", initialfile=default,
            initialdir=str(ROOT_DIR), filetypes=[("텍스트 파일", "*.txt")],
        )
        if not path:
            return
        try:
            Path(path).write_text(content + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("저장 실패", str(exc))
            return
        self._set_status(f"저장됨: {path}")

    def _copy_text(self) -> None:
        content = self._committed_text()
        if not content:
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self._set_status("전체 내용을 클립보드에 복사했습니다")

    def _clear_text(self) -> None:
        if self.text.get("1.0", "end").strip() and not messagebox.askyesno(
            "지우기", "받아쓴 내용을 모두 지울까요?"
        ):
            return
        self.text.delete("1.0", "end")

    def _on_close(self) -> None:
        if self.listening:
            self._stop_listening()
        else:
            self.segment_queue.put(None)
            self.partial_queue.put(None)
        self.settings.interim_enabled = self.interim_var.get()
        self.settings.show_timestamp = self.ts_var.get()
        self.settings.autoscroll = self.scroll_var.get()
        self.settings.model_name = self._selected_model()
        self.settings.language = self._selected_language()
        self.settings.save()
        self.destroy()


def run() -> None:
    SttApp().mainloop()
