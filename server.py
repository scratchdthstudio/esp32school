#!/usr/bin/env python3
"""
ESP32-CAM 녹화 서버 (Flask, 웹사이트 기반) + 5분 단위 사람 수 카운팅

흐름
  1) 웹 페이지에서 [촬영 시작] -> 서버가 세션을 만들고 "record" 명령을 대기시킴
  2) ESP32-CAM 이 /api/device/poll 로 명령을 받아 촬영 시작, 프레임(JPEG)을 계속 /api/device/frame 으로 전송
  3) 웹 페이지에서 [촬영 종료] -> 다음 프레임 응답에 "stop" 이 실려 ESP32-CAM 이 멈춤
  4) 서버가 임시 저장된 JPEG 들을 하나의 mp4(H.264)로 합친다
  5) (추가) 같은 JPEG 들에서 사람을 검출해 '프레임 수신 시각' 기준 5분 단위(14:00~14:05 ...)로 집계하고
     결과를 recordings/<세션>.json 에 저장, 그 뒤 임시 파일을 삭제

사람 검출기 (먼저 되는 것을 사용)
  - ultralytics YOLO (권장, 정확):  pip install ultralytics      -> 평균/최대/고유 인원(추적) 모두 제공
  - OpenCV HOG (대체, 정확도 낮음): pip install opencv-python    -> 평균/최대만 제공

실행:  python server.py            (기본 0.0.0.0:5000)
환경변수:
  PORT                  서버 포트
  KEEP_FRAMES=1         임시 JPEG 삭제 안 함
  PEOPLE_COUNT=0        사람 카운팅 끄기
  PEOPLE_MODEL          YOLO 모델 (기본 yolov8n.pt, 처음 실행 시 자동 다운로드)
  PEOPLE_SAMPLE_FPS     초당 분석할 프레임 수 (기본 2, CPU 부담 줄이려면 낮추기)
  PEOPLE_CONF           검출 신뢰도 임계값 (기본 0.4)
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

BASE_DIR = Path(__file__).resolve().parent
REC_DIR = BASE_DIR / "recordings"
TMP_DIR = BASE_DIR / "tmp_frames"
REC_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

KEEP_FRAMES = os.environ.get("KEEP_FRAMES") == "1"
MAX_GAP_SEC = 2.0          # 프레임 사이 공백(Wi-Fi 끊김 등)이 이보다 길면 영상에서는 이 길이로 줄임
DEVICE_ONLINE_SEC = 5.0    # 이 시간 안에 장치 통신이 있으면 '연결됨'
FRAMESIZES = ["QQVGA", "QVGA", "CIF", "VGA", "SVGA", "XGA", "SXGA", "UXGA"]
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.mp4$")

# ---- 사람 카운팅 설정
PEOPLE_COUNT = os.environ.get("PEOPLE_COUNT", "1") != "0"
PEOPLE_MODEL = os.environ.get("PEOPLE_MODEL", "yolov8n.pt")
PEOPLE_SAMPLE_FPS = max(0.1, float(os.environ.get("PEOPLE_SAMPLE_FPS", "2")))
PEOPLE_CONF = float(os.environ.get("PEOPLE_CONF", "0.4"))
BUCKET_MIN = 5             # 집계 단위(분)

# ESP32 의 HTTPClient 가 연결을 재사용(keep-alive)할 수 있도록 HTTP/1.1 사용
WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

lock = threading.RLock()
state = {
    "recording": False,
    "session": None,
    "dir": None,
    "frames": [],          # [(파일명, 수신시각)]
    "counter": 0,
    "size": None,          # (w, h) 첫 프레임 기준
    "started": 0.0,
    "bytes": 0,
    "skipped": 0,
    "last_frame": None,
    "settings": {"framesize": "VGA", "quality": 12, "interval_ms": 0},
    "device": {"seen": 0.0, "ip": None, "rssi": None},
}
jobs = {}  # session -> {"state": "encoding"|"analyzing"|"done"|"error"|"no_frames", ...}


# ---------------------------------------------------------------- 유틸
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def jpeg_size(b):
    """JPEG 헤더(SOF)에서 (폭, 높이)를 읽는다. JPEG 가 아니면 None."""
    n = len(b)
    if n < 4 or b[0] != 0xFF or b[1] != 0xD8:
        return None
    i = 2
    while i + 9 < n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9:
            return None
        seglen = (b[i + 2] << 8) | b[i + 3]
        if m in (0xC0, 0xC1, 0xC2):
            return ((b[i + 7] << 8) | b[i + 8], (b[i + 5] << 8) | b[i + 6])
        i += 2 + seglen
    return None


def touch_device():
    with lock:
        d = state["device"]
        d["seen"] = time.time()
        d["ip"] = request.remote_addr
        rssi = request.args.get("rssi")
        if rssi is not None:
            d["rssi"] = rssi


def text(body):
    return Response(body, mimetype="text/plain")


# ---------------------------------------------------------------- 사람 카운팅
_det = {"loaded": False, "method": None, "obj": None}
_det_lock = threading.Lock()
_analysis_lock = threading.Lock()   # 분석은 한 번에 하나씩 (추적기 상태 보호, CPU 보호)


def get_detector():
    """검출기를 처음 한 번만 로드. YOLO -> OpenCV HOG 순으로 시도."""
    with _det_lock:
        if _det["loaded"]:
            return _det
        _det["loaded"] = True
        if not PEOPLE_COUNT:
            return _det
        try:
            from ultralytics import YOLO
            _det.update(method="yolo", obj=YOLO(PEOPLE_MODEL))
            print(f"[사람 카운팅] YOLO 사용 ({PEOPLE_MODEL})")
            return _det
        except Exception as e:  # noqa: BLE001
            print(f"[사람 카운팅] YOLO 사용 불가 ({e.__class__.__name__}), OpenCV HOG 로 대체 시도")
        try:
            import cv2
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            _det.update(method="hog", obj=hog)
            print("[사람 카운팅] OpenCV HOG 사용 (정확도 낮음, 고유 인원 집계 불가)")
        except Exception:  # noqa: BLE001
            print("[경고] 사람 카운팅 불가: 'pip install ultralytics' 또는 'pip install opencv-python' 필요")
        return _det


def bucket_start(t):
    """수신 시각 t(epoch) 를 5분 경계(예: 14:05:00)로 내린 datetime."""
    dt = datetime.fromtimestamp(t)
    return dt.replace(minute=dt.minute // BUCKET_MIN * BUCKET_MIN, second=0, microsecond=0)


def summarize_buckets(per_sample, method):
    """per_sample: [(수신시각, 인원수, 추적ID 리스트 또는 None)] -> 5분 단위 요약.
    표본이 없는 구간(통신 끊김 등)도 samples=0 으로 채워 넣는다."""
    if not per_sample:
        return {"buckets": [], "total_unique": None, "peak": 0}
    acc = {}
    all_ids = set()
    for t, n, ids in per_sample:
        b = acc.setdefault(bucket_start(t), {"samples": 0, "sum": 0, "max": 0, "ids": set()})
        b["samples"] += 1
        b["sum"] += n
        b["max"] = max(b["max"], n)
        if ids:
            b["ids"].update(ids)
            all_ids.update(ids)

    step = timedelta(minutes=BUCKET_MIN)
    cur, last = min(acc), max(acc)
    out = []
    while cur <= last:
        b = acc.get(cur)
        item = {"start": cur.strftime("%H:%M"), "end": (cur + step).strftime("%H:%M"),
                "date": cur.strftime("%Y-%m-%d")}
        if b:
            item.update(samples=b["samples"], avg=round(b["sum"] / b["samples"], 2), max=b["max"],
                        unique=len(b["ids"]) if method == "yolo" else None)
        else:
            item.update(samples=0, avg=None, max=None, unique=None)
        out.append(item)
        cur += step
    return {
        "buckets": out,
        "total_unique": len(all_ids) if method == "yolo" else None,
        "peak": max((n for _, n, _ in per_sample), default=0),
    }


def analyze_people(d, frames):
    """저장된 JPEG 들에서 사람을 검출해 5분 단위로 집계. 불가능하면 None."""
    det = get_detector()
    method = det["method"]
    if not method:
        return None

    # 프레임이 너무 많으면 CPU 가 버티지 못하므로 PEOPLE_SAMPLE_FPS 간격으로 솎아서 분석
    gap = 1.0 / PEOPLE_SAMPLE_FPS
    samples, last = [], None
    for name, t in frames:
        if last is None or t - last >= gap:
            samples.append((name, t))
            last = t

    per_sample = []
    with _analysis_lock:
        first = True
        for name, t in samples:
            path = str(d / name)
            ids = None
            try:
                if method == "yolo":
                    # persist=False 인 첫 프레임에서 추적기가 새로 시작된다 (세션 간 ID 섞임 방지)
                    r = det["obj"].track(path, persist=not first, classes=[0], conf=PEOPLE_CONF,
                                         verbose=False)[0]
                    n = len(r.boxes)
                    ids = r.boxes.id.int().tolist() if (n and r.boxes.id is not None) else []
                else:
                    import cv2
                    import numpy as np
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    if w > 640:
                        img = cv2.resize(img, (640, int(h * 640 / w)))
                    _, weights = det["obj"].detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    n = sum(1 for x in np.ravel(weights) if x >= 0.5)
            except Exception:  # noqa: BLE001
                continue
            first = False
            per_sample.append((t, n, ids))

    res = summarize_buckets(per_sample, method)
    res.update(method=method, bucket_minutes=BUCKET_MIN, sample_fps=PEOPLE_SAMPLE_FPS,
               analyzed_frames=len(per_sample))
    return res


# ---------------------------------------------------------------- mp4 생성
def encode_session(snap):
    sid, d, frames = snap["session"], snap["dir"], snap["frames"]
    n = len(frames)

    def set_job(**kw):
        with lock:
            jobs[sid] = {**jobs.get(sid, {}), **kw}

    if n == 0:
        set_job(state="no_frames")
        shutil.rmtree(d, ignore_errors=True)
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        set_job(state="error", message="ffmpeg 를 찾을 수 없습니다 (pip install imageio-ffmpeg)")
        return

    span = frames[-1][1] - frames[0][1]
    fps = n / span if (n > 1 and span > 0) else 10.0
    fps = max(1, min(30, round(fps)))

    lines = ["ffconcat version 1.0"]
    for i, (name, t) in enumerate(frames):
        if i + 1 < n:
            dur = min(max(frames[i + 1][1] - t, 0.001), MAX_GAP_SEC)
        else:
            dur = 1.0 / fps
        lines += [f"file '{name}'", f"duration {dur:.4f}"]
    lines.append(f"file '{frames[-1][0]}'")  # ffmpeg concat: 마지막 프레임 duration 적용용
    (d / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = REC_DIR / f"{sid}.mp4"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", "list.txt",
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart", str(out),
    ]
    try:
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=3600)
    except Exception as e:  # noqa: BLE001
        set_job(state="error", message=str(e))
        return
    if r.returncode != 0 or not out.exists():
        set_job(state="error", message=(r.stderr or "ffmpeg 실패")[-300:])
        return

    meta = {
        "session": sid,
        "started": datetime.fromtimestamp(frames[0][1]).isoformat(timespec="seconds"),
        "ended": datetime.fromtimestamp(frames[-1][1]).isoformat(timespec="seconds"),
        "frames": n,
        "fps": fps,
        "real_duration_sec": round(span, 2),
        "size_bytes": out.stat().st_size,
        "people": None,
    }

    # mp4 가 만들어졌으니 먼저 영상은 쓸 수 있게 하고, 이어서 사람 카운팅 (임시 JPEG 삭제 전에)
    (REC_DIR / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if PEOPLE_COUNT:
        set_job(state="analyzing", file=out.name, frames=n, fps=fps)
        try:
            meta["people"] = analyze_people(d, frames)
        except Exception as e:  # noqa: BLE001
            meta["people"] = {"error": str(e)[:200]}
        (REC_DIR / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not KEEP_FRAMES:
        shutil.rmtree(d, ignore_errors=True)
    people = meta["people"] or {}
    set_job(state="done", file=out.name, frames=n, fps=fps,
            people_peak=people.get("peak"), people_unique=people.get("total_unique"))


# ---------------------------------------------------------------- 웹 UI용 API
@app.post("/api/record/start")
def api_start():
    data = request.get_json(silent=True) or {}
    with lock:
        if state["recording"]:
            return jsonify(error="이미 촬영 중입니다"), 409
        cur = state["settings"]
        fs = str(data.get("framesize", cur["framesize"])).upper()
        if fs not in FRAMESIZES:
            return jsonify(error="지원하지 않는 해상도"), 400
        try:
            q = max(10, min(63, int(data.get("quality", cur["quality"]))))
            iv = max(0, min(5000, int(data.get("interval_ms", cur["interval_ms"]))))
        except (TypeError, ValueError):
            return jsonify(error="숫자 값이 올바르지 않습니다"), 400
        state["settings"] = {"framesize": fs, "quality": q, "interval_ms": iv}

        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = TMP_DIR / sid
        d.mkdir(parents=True, exist_ok=True)
        state.update(recording=True, session=sid, dir=d, frames=[], counter=0, size=None,
                     started=time.time(), bytes=0, skipped=0, last_frame=None)
    return jsonify(ok=True, session=sid)


@app.post("/api/record/stop")
def api_stop():
    with lock:
        if not state["recording"]:
            return jsonify(error="촬영 중이 아닙니다"), 409
        snap = {"session": state["session"], "dir": state["dir"], "frames": list(state["frames"])}
        state.update(recording=False, session=None, last_frame=None)
        jobs[snap["session"]] = {"state": "encoding", "frames": len(snap["frames"])}
    threading.Thread(target=encode_session, args=(snap,), daemon=True).start()
    return jsonify(ok=True, session=snap["session"], frames=len(snap["frames"]))


@app.get("/api/status")
def api_status():
    now = time.time()
    with lock:
        fr = state["frames"]
        n = len(fr)
        fps = 0.0
        if n > 1 and fr[-1][1] > fr[0][1]:
            fps = (n - 1) / (fr[-1][1] - fr[0][1])
        dev = state["device"]
        return jsonify(
            recording=state["recording"],
            session=state["session"],
            frames=n,
            skipped=state["skipped"],
            bytes=state["bytes"],
            elapsed=(now - state["started"]) if state["recording"] else 0,
            fps=round(fps, 1),
            settings=state["settings"],
            device_online=(now - dev["seen"]) < DEVICE_ONLINE_SEC,
            device_ip=dev["ip"],
            device_rssi=dev["rssi"],
            jobs=[{"session": k, **v} for k, v in list(jobs.items())[-5:]],
        )


@app.get("/api/preview.jpg")
def api_preview():
    with lock:
        data = state["last_frame"]
    if not data:
        return Response(status=404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/recordings")
def api_recordings():
    items = []
    for p in sorted(REC_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        meta = {}
        mp = p.with_suffix(".json")
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        people = meta.get("people") or {}
        items.append({
            "name": p.name,
            "size": p.stat().st_size,
            "started": meta.get("started"),
            "ended": meta.get("ended"),
            "frames": meta.get("frames"),
            "fps": meta.get("fps"),
            "has_people": bool(people.get("buckets")),
            "people_peak": people.get("peak"),
        })
    return jsonify(items)


@app.get("/api/recordings/<name>/people")
def api_people(name):
    """해당 영상의 5분 단위 사람 수 통계."""
    if not NAME_RE.match(name):
        return Response(status=404)
    mp = (REC_DIR / name).with_suffix(".json")
    if not mp.exists():
        return jsonify(error="통계 파일이 없습니다"), 404
    try:
        people = json.loads(mp.read_text(encoding="utf-8")).get("people")
    except Exception:  # noqa: BLE001
        return jsonify(error="통계 파일을 읽을 수 없습니다"), 500
    if not people:
        return jsonify(error="사람 카운팅 결과가 없습니다 (검출기 미설치 또는 카운팅 꺼짐)"), 404
    if people.get("error"):
        return jsonify(error="분석 중 오류: " + people["error"]), 500
    return jsonify(people)


@app.get("/recordings/<name>")
def get_recording(name):
    if not NAME_RE.match(name):
        return Response(status=404)
    return send_from_directory(REC_DIR, name, as_attachment=request.args.get("dl") == "1")


@app.delete("/api/recordings/<name>")
def delete_recording(name):
    if not NAME_RE.match(name):
        return Response(status=404)
    p = REC_DIR / name
    if not p.exists():
        return Response(status=404)
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)
    return jsonify(ok=True)


# ---------------------------------------------------------------- ESP32-CAM용 API
@app.get("/api/device/poll")
def device_poll():
    """대기 중인 ESP32-CAM 이 1초마다 호출. 촬영 중이면 record 명령과 설정을 돌려준다."""
    touch_device()
    with lock:
        if state["recording"]:
            s = state["settings"]
            return text(
                f"cmd=record&session={state['session']}&framesize={s['framesize']}"
                f"&quality={s['quality']}&interval={s['interval_ms']}"
            )
    return text("cmd=idle")


@app.post("/api/device/frame")
def device_frame():
    """촬영 중인 ESP32-CAM 이 JPEG 한 장씩 전송. 응답으로 계속(record)/중지(stop) 명령을 준다."""
    touch_device()
    sid = request.args.get("session", "")
    data = request.get_data()

    with lock:
        if not (state["recording"] and state["session"] == sid):
            return text("cmd=stop")
        size = jpeg_size(data)
        if size is None:
            state["skipped"] += 1
            return text("cmd=record")
        if state["size"] is None:
            state["size"] = size
        elif state["size"] != size:       # 해상도 변경 직후의 잔여 프레임 등
            state["skipped"] += 1
            return text("cmd=record")
        idx = state["counter"]
        state["counter"] += 1
        d = state["dir"]

    name = f"{idx:06d}.jpg"
    try:
        (d / name).write_bytes(data)
    except OSError:
        return text("cmd=stop")
    t = time.time()

    with lock:
        if state["recording"] and state["session"] == sid:
            state["frames"].append((name, t))
            state["bytes"] += len(data)
            state["last_frame"] = data
            return text("cmd=record")
    (d / name).unlink(missing_ok=True)    # 처리 중에 종료된 경우
    return text("cmd=stop")


# ---------------------------------------------------------------- 웹 페이지
INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-CAM 녹화 서버</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --fg:#1d2330; --muted:#6b7385; --line:#e2e5eb; --acc:#2f6fed; --red:#d93a3a; --green:#1f9d55; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151b; --card:#1b2029; --fg:#e8ebf2; --muted:#98a1b5; --line:#2b3240; --acc:#6b9bff; --red:#ff6b6b; --green:#4cc27f; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif; }
  main { max-width:960px; margin:0 auto; padding:20px 16px 48px; }
  h1 { font-size:20px; margin:0 0 16px; }
  h2 { font-size:15px; margin:0 0 12px; color:var(--muted); font-weight:600; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px; }
  .row { display:flex; flex-wrap:wrap; gap:12px; align-items:end; }
  label { display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--muted); }
  select,input { background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:inherit; min-width:110px; }
  button { border:0; border-radius:8px; padding:9px 16px; font:inherit; font-weight:600; cursor:pointer; color:#fff; background:var(--acc); }
  button.stop { background:var(--red); }
  button.ghost { background:transparent; color:var(--fg); border:1px solid var(--line); font-weight:500; padding:5px 10px; font-size:13px; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
  .chip { font-size:13px; padding:3px 10px; border-radius:99px; border:1px solid var(--line); color:var(--muted); }
  .chip.on { color:var(--green); border-color:var(--green); }
  .chip.rec { color:var(--red); border-color:var(--red); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:10px; margin-top:14px; }
  .stat b { display:block; font-size:20px; font-variant-numeric:tabular-nums; }
  .stat span { font-size:12px; color:var(--muted); }
  #preview { width:100%; max-height:420px; object-fit:contain; background:#000; border-radius:8px; margin-top:14px; display:none; }
  .rec-item { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; padding:10px 0; border-top:1px solid var(--line); }
  .rec-item:first-child { border-top:0; }
  .rec-name { font-weight:600; }
  .rec-meta { font-size:12px; color:var(--muted); }
  .rec-actions { display:flex; gap:6px; }
  video { width:100%; max-height:480px; background:#000; border-radius:8px; margin-top:10px; display:none; }
  .msg { font-size:13px; color:var(--muted); margin-top:10px; min-height:1.4em; }
  a.btnlink { text-decoration:none; display:inline-block; }
  table.pt { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; margin-top:8px; }
  table.pt th, table.pt td { text-align:right; padding:6px 8px; border-top:1px solid var(--line); }
  table.pt th:first-child, table.pt td:first-child { text-align:left; }
  table.pt th { color:var(--muted); font-weight:600; border-top:0; }
  .bar { display:inline-block; height:8px; background:var(--acc); border-radius:4px; vertical-align:middle; margin-left:6px; }
</style>
</head>
<body>
<main>
  <h1>ESP32-CAM 녹화 서버</h1>

  <div class="card">
    <div class="chips">
      <span class="chip" id="chipDev">장치: 확인 중</span>
      <span class="chip" id="chipRec">대기</span>
    </div>
    <div class="row">
      <label>해상도
        <select id="framesize">
          <option>QQVGA</option><option>QVGA</option><option>CIF</option>
          <option selected>VGA</option><option>SVGA</option><option>XGA</option>
          <option>SXGA</option><option>UXGA</option>
        </select>
      </label>
      <label>JPEG 품질 (10~63, 낮을수록 고화질)
        <input id="quality" type="number" min="10" max="63" value="12">
      </label>
      <label>최소 프레임 간격 ms (0=최대 속도)
        <input id="interval" type="number" min="0" max="5000" value="0">
      </label>
      <button id="btnStart">촬영 시작</button>
      <button id="btnStop" class="stop" disabled>촬영 종료</button>
    </div>
    <div class="msg" id="msg"></div>

    <div class="stats">
      <div class="stat"><b id="sFrames">0</b><span>수신 프레임</span></div>
      <div class="stat"><b id="sFps">0</b><span>평균 FPS</span></div>
      <div class="stat"><b id="sElapsed">00:00</b><span>경과 시간</span></div>
      <div class="stat"><b id="sSize">0 MB</b><span>임시 저장 용량</span></div>
    </div>
    <img id="preview" alt="라이브 미리보기">
  </div>

  <div class="card">
    <h2>mp4 변환 · 사람 카운팅 상태</h2>
    <div id="jobs" class="rec-meta">없음</div>
  </div>

  <div class="card">
    <h2>저장된 영상</h2>
    <div id="recs" class="rec-meta">없음</div>
    <video id="player" controls></video>
  </div>

  <div class="card" id="peopleCard" style="display:none">
    <h2 id="peopleTitle">5분 단위 인원 통계</h2>
    <div id="peopleBody" class="rec-meta"></div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
let recording = false, settingsFilled = false;

const fmtTime = s => { s = Math.floor(s); return String(Math.floor(s/60)).padStart(2,'0') + ':' + String(s%60).padStart(2,'0'); };
const fmtMB = b => (b/1048576).toFixed(1) + ' MB';

async function api(path, opts) {
  const r = await fetch(path, opts);
  let j = {}; try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

async function refresh() {
  try {
    const s = await api('/api/status');
    recording = s.recording;
    if (!settingsFilled) {
      $('framesize').value = s.settings.framesize; $('quality').value = s.settings.quality; $('interval').value = s.settings.interval_ms;
      settingsFilled = true;
    }
    const dev = $('chipDev');
    dev.className = 'chip' + (s.device_online ? ' on' : '');
    dev.textContent = s.device_online
      ? '장치 연결됨 ' + (s.device_ip || '') + (s.device_rssi ? ' (' + s.device_rssi + ' dBm)' : '')
      : '장치 오프라인';
    const rc = $('chipRec');
    rc.className = 'chip' + (s.recording ? ' rec' : '');
    rc.textContent = s.recording ? (s.frames === 0 ? '장치 응답 대기 중…' : '● 촬영 중 ' + s.session) : '대기';
    $('btnStart').disabled = s.recording; $('btnStop').disabled = !s.recording;
    ['framesize','quality','interval'].forEach(id => $(id).disabled = s.recording);
    $('sFrames').textContent = s.frames; $('sFps').textContent = s.fps;
    $('sElapsed').textContent = fmtTime(s.elapsed); $('sSize').textContent = fmtMB(s.bytes);
    $('preview').style.display = s.recording ? 'block' : 'none';

    const names = {encoding:'mp4 변환 중…', analyzing:'mp4 완료 · 사람 카운팅 중…', done:'완료', error:'오류', no_frames:'수신된 프레임 없음'};
    $('jobs').innerHTML = s.jobs.length
      ? s.jobs.slice().reverse().map(j => `${j.session} — ${names[j.state] || j.state}` + (j.frames ? ` (${j.frames}장)` : '')
          + (j.state === 'done' && j.people_peak != null ? ` · 최대 동시 ${j.people_peak}명` : '')
          + (j.message ? ` : ${j.message}` : '')).join('<br>')
      : '없음';
    if (s.jobs.some(j => j.state === 'done' || j.state === 'analyzing')) loadRecs();
  } catch (e) { $('chipDev').textContent = '서버 연결 끊김'; $('chipDev').className = 'chip'; }
}

let lastRecKey = '';
async function loadRecs() {
  const list = await api('/api/recordings');
  const key = JSON.stringify(list.map(x => x.name + x.size + x.has_people));
  if (key === lastRecKey) return; lastRecKey = key;
  $('recs').innerHTML = list.length ? '' : '없음';
  list.forEach(r => {
    const d = document.createElement('div'); d.className = 'rec-item';
    d.innerHTML = `<div><div class="rec-name">${r.name}</div><div class="rec-meta">${r.started ? r.started.replace('T',' ') + ' ~ ' + r.ended.slice(11) + ' · ' : ''}${r.frames ? r.frames + '장 · ' : ''}${fmtMB(r.size)}${r.has_people ? ' · 최대 동시 ' + r.people_peak + '명' : ''}</div></div>
      <div class="rec-actions"><button class="ghost" data-act="play">재생</button>${r.has_people ? '<button class="ghost" data-act="people">인원 통계</button>' : ''}<a class="btnlink" href="/recordings/${r.name}?dl=1"><button class="ghost">다운로드</button></a><button class="ghost" data-act="del">삭제</button></div>`;
    d.querySelector('[data-act=play]').onclick = () => { const p = $('player'); p.src = '/recordings/' + r.name; p.style.display = 'block'; p.play(); p.scrollIntoView({behavior:'smooth', block:'center'}); };
    const pb = d.querySelector('[data-act=people]');
    if (pb) pb.onclick = () => showPeople(r.name);
    d.querySelector('[data-act=del]').onclick = async () => { if (!confirm(r.name + ' 을(를) 삭제할까요?')) return; await api('/api/recordings/' + r.name, {method:'DELETE'}); lastRecKey = ''; $('peopleCard').style.display = 'none'; loadRecs(); };
    $('recs').appendChild(d);
  });
}

async function showPeople(name) {
  const card = $('peopleCard'), body = $('peopleBody');
  card.style.display = 'block'; $('peopleTitle').textContent = '5분 단위 인원 통계 — ' + name;
  body.textContent = '불러오는 중…';
  try {
    const p = await api('/api/recordings/' + name + '/people');
    const hasUniq = p.method === 'yolo';
    const maxAvg = Math.max(1, ...p.buckets.map(b => b.avg || 0));
    let h = `<div>검출기: ${p.method === 'yolo' ? 'YOLO(추적 사용)' : 'OpenCV HOG(정확도 낮음, 고유 인원 없음)'} · 초당 ${p.sample_fps}프레임 분석 · 분석 ${p.analyzed_frames}장`
          + ` · 최대 동시 ${p.peak}명` + (hasUniq && p.total_unique != null ? ` · 전체 고유 약 ${p.total_unique}명` : '') + '</div>';
    h += '<table class="pt"><tr><th>시간대</th><th>평균 인원</th><th>최대 동시</th>' + (hasUniq ? '<th>고유 인원</th>' : '') + '<th>분석 프레임</th></tr>';
    p.buckets.forEach(b => {
      const w = b.avg ? Math.round(b.avg / maxAvg * 80) : 0;
      h += `<tr><td>${b.date.slice(5)} ${b.start}~${b.end}</td><td>${b.avg == null ? '데이터 없음' : b.avg}<span class="bar" style="width:${w}px"></span></td><td>${b.max == null ? '-' : b.max}</td>`
         + (hasUniq ? `<td>${b.unique == null ? '-' : b.unique}</td>` : '') + `<td>${b.samples}</td></tr>`;
    });
    body.innerHTML = h + '</table>';
    card.scrollIntoView({behavior:'smooth', block:'center'});
  } catch (e) { body.textContent = e.message; }
}

$('btnStart').onclick = async () => {
  $('msg').textContent = '';
  try {
    await api('/api/record/start', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ framesize: $('framesize').value, quality: +$('quality').value, interval_ms: +$('interval').value }) });
  } catch (e) { $('msg').textContent = e.message; }
  refresh();
};
$('btnStop').onclick = async () => {
  try { await api('/api/record/stop', { method:'POST' }); $('msg').textContent = '촬영을 종료했습니다. mp4 로 변환하고 사람 수를 집계합니다.'; }
  catch (e) { $('msg').textContent = e.message; }
  refresh();
};

function previewLoop() {
  if (!recording) return setTimeout(previewLoop, 500);
  const im = new Image();
  im.onload = () => { $('preview').src = im.src; setTimeout(previewLoop, 150); };
  im.onerror = () => setTimeout(previewLoop, 500);
  im.src = '/api/preview.jpg?t=' + Date.now();
}

setInterval(refresh, 1000); refresh(); loadRecs(); previewLoop();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    if not find_ffmpeg():
        print("[경고] ffmpeg 를 찾지 못했습니다. 'pip install imageio-ffmpeg' 또는 ffmpeg 를 설치하세요.")
    if PEOPLE_COUNT:
        threading.Thread(target=get_detector, daemon=True).start()   # 모델을 미리 로드/다운로드
    print(f"웹 페이지: http://<이 PC의 IP>:{port}/")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
