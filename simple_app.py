"""Lightweight face check: capture, describe the face, capture again, same person?

    .venv\\Scripts\\python.exe simple_app.py        ->  http://127.0.0.1:8080

The browser version of ``Face_detect.ipynb``: no roster, no attendance, nothing
stored. One page, two captures.

    1. Capture a photo  -> DeepFace.analyze: face box, age, gender, race, emotion
    2. Capture another  -> DeepFace.verify:  same person or not, with the
                           distance, threshold, model and face areas

Frames stay in memory on both sides; the server keeps no state between calls,
so the browser sends both photos for the verification.

Endpoints
    GET  /          the page
    POST /analyze   image            -> attributes of the largest face
    POST /verify    image1, image2   -> DeepFace.verify result
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

MODEL = "Facenet"  # weights already cached locally; the notebook's VGG-Face default is ~580 MB more
DETECTOR = "opencv"
ACTIONS = ("age", "gender", "race", "emotion")

app = FastAPI(title="Simple Face Check")
_lock = threading.Lock()  # DeepFace/TensorFlow is not safe to call concurrently


def _decode(payload: bytes) -> np.ndarray:
    import cv2

    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("not an image")
    return frame


def _round(d: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(v), 1) for k, v in d.items()}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, Any]:
    """Attributes of the largest face, as the notebook's first step printed them."""
    from deepface import DeepFace

    try:
        frame = _decode(await image.read())
    except ValueError:
        return {"ok": False, "message": "bad image"}

    started = time.perf_counter()
    with _lock:
        faces = DeepFace.analyze(
            img_path=frame, actions=ACTIONS, detector_backend=DETECTOR,
            enforce_detection=False, silent=True,
        )
    faces = [f for f in faces if float(f.get("face_confidence") or 0) > 0]
    if not faces:
        return {"ok": False, "message": "no face detected", "frame_size": [frame.shape[1], frame.shape[0]]}

    f = max(faces, key=lambda r: r["region"]["w"] * r["region"]["h"])
    r = f["region"]
    return {
        "ok": True,
        "faces": len(faces),
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "region": [int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])],
        "face_confidence": round(float(f["face_confidence"]), 3),
        "age": int(f["age"]),
        "gender": f["dominant_gender"],
        "gender_scores": _round(f["gender"]),
        "race": f["dominant_race"],
        "race_scores": _round(f["race"]),
        "emotion": f["dominant_emotion"],
        "emotion_scores": _round(f["emotion"]),
        "seconds": round(time.perf_counter() - started, 2),
    }


@app.post("/verify")
async def verify(image1: UploadFile = File(...), image2: UploadFile = File(...)) -> dict[str, Any]:
    """Same person? The notebook's DeepFace.verify, returned in full."""
    from deepface import DeepFace

    try:
        a = _decode(await image1.read())
        b = _decode(await image2.read())
    except ValueError:
        return {"ok": False, "message": "bad image"}

    try:
        with _lock:
            res = DeepFace.verify(
                img1_path=a, img2_path=b, model_name=MODEL, detector_backend=DETECTOR,
                distance_metric="cosine", enforce_detection=True, silent=True,
            )
    except ValueError as exc:  # enforce_detection: one of the photos has no face
        return {"ok": False, "message": str(exc).split(".")[0]}

    areas = res.get("facial_areas") or {}
    return {
        "ok": True,
        "verified": bool(res["verified"]),
        "distance": round(float(res["distance"]), 4),
        "threshold": float(res["threshold"]),
        "model": res.get("model"),
        "detector": res.get("detector_backend"),
        "metric": res.get("similarity_metric"),
        "seconds": round(float(res.get("time", 0)), 2),
        "area1": [areas.get("img1", {}).get(k) for k in ("x", "y", "w", "h")],
        "area2": [areas.get("img2", {}).get(k) for k in ("x", "y", "w", "h")],
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Simple Face Check</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0 auto;padding:16px;max-width:960px}
  h1{font-size:20px;margin:8px 0 12px}.muted{color:#888;font-size:13px}
  video{display:block;width:100%;max-width:560px;aspect-ratio:4/3;object-fit:cover;background:#000;border-radius:10px}
  button{background:#0284c7;color:#fff;border:0;border-radius:8px;padding:9px 14px;margin:8px 6px 0 0;font-size:14px;cursor:pointer}
  button.ghost{background:transparent;border:1px solid #555}button:disabled{opacity:.4}
  #msg{margin:10px 0;min-height:22px;font-weight:600}#msg.bad{color:#f87171}
  .shots{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
  .shot{background:#1a1a1a;border-radius:10px;padding:10px}.shot h3{margin:0 0 8px;font-size:14px;color:#bbb}
  canvas{width:100%;border-radius:6px;background:#000;display:block}
  dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:13px;margin:8px 0 0}dt{color:#888}dd{margin:0}
  .bar{display:inline-block;height:8px;background:#0284c7;border-radius:4px;vertical-align:middle;margin-right:6px}
  #verdict{margin-top:14px;padding:14px;border-radius:10px;font-size:22px;font-weight:700;text-align:center;display:none}
  #verdict.same{background:#064e3b;color:#6ee7b7;display:block}#verdict.diff{background:#7f1d1d;color:#fca5a5;display:block}
  #vdetails{font-size:13px;font-weight:400;margin-top:6px;color:#ddd}
</style></head><body>
<h1>Simple Face Check <span class="muted">· capture, describe, capture again, same person?</span></h1>
<video id="cam" autoplay playsinline muted></video>
<div>
  <button id="start">Start camera</button>
  <button id="shot1" disabled>Capture 1st photo</button>
  <button id="shot2" disabled>Capture 2nd photo</button>
  <button class="ghost" id="reset" disabled>Reset</button>
</div>
<div id="msg"></div>
<div id="verdict"><div id="vtext"></div><div id="vdetails"></div></div>
<div class="shots">
  <div class="shot"><h3>First photo</h3><canvas id="c1" width="480" height="360"></canvas><dl id="d1"></dl></div>
  <div class="shot"><h3>Second photo</h3><canvas id="c2" width="480" height="360"></canvas><dl id="d2"></dl></div>
</div>
<script>
const $=s=>document.querySelector(s), video=$('#cam'), msg=$('#msg');
const shots={1:null,2:null};
function say(t,bad){msg.textContent=t;msg.className=bad?'bad':''}
function grab(n){const c=$('#c'+n),s=480/video.videoWidth;c.width=480;c.height=Math.round(video.videoHeight*s);
  c.getContext('2d').drawImage(video,0,0,c.width,c.height);return new Promise(r=>c.toBlob(r,'image/jpeg',0.9))}
function box(n,[x,y,w,h],color,above,below){const ctx=$('#c'+n).getContext('2d');ctx.lineWidth=3;ctx.strokeStyle=color;ctx.strokeRect(x,y,w,h);
  ctx.font='bold 14px system-ui';for(const [t,yy] of [[above,y-8],[below,y+h+18]]){if(!t)continue;const tw=ctx.measureText(t).width+8;
  ctx.fillStyle='rgba(255,255,255,.85)';ctx.fillRect(x,yy-14,tw,18);ctx.fillStyle=color;ctx.fillText(t,x+4,yy)}}
const pct=o=>Object.entries(o).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span class="bar" style="width:${Math.max(2,v*0.6)}px"></span>${k} ${v}%`).join('<br>');
function details(n,a){$('#d'+n).innerHTML=`<dt>Face</dt><dd>at [${a.region.join(', ')}] · confidence ${a.face_confidence}${a.faces>1?` · ${a.faces} faces, largest used`:''}</dd>
  <dt>Age</dt><dd>${a.age}</dd><dt>Gender</dt><dd>${a.gender}<br>${pct(a.gender_scores)}</dd>
  <dt>Race</dt><dd>${a.race}<br>${pct(a.race_scores)}</dd><dt>Emotion</dt><dd>${a.emotion}<br>${pct(a.emotion_scores)}</dd>
  <dt>Time</dt><dd>${a.seconds}s</dd>`}
async function capture(n){setBusy(true);say(`analysing photo ${n}…`);
  try{const blob=await grab(n);shots[n]=blob;const f=new FormData();f.append('image',blob,'p.jpg');
    const a=await (await fetch('/analyze',{method:'POST',body:f})).json();
    if(!a.ok){say(a.message,true);$('#d'+n).innerHTML='';return}
    box(n,a.region,'#ef4444',`Age: ${a.age}, ${a.gender}`,`Emotion: ${a.emotion}`);details(n,a);say(`photo ${n}: ${a.gender}, ~${a.age}, ${a.emotion}`);
    if(n===2&&shots[1])await verify()}
  catch(e){say(String(e),true)}finally{setBusy(false)}}
async function verify(){say('verifying…');const f=new FormData();f.append('image1',shots[1],'a.jpg');f.append('image2',shots[2],'b.jpg');
  const v=await (await fetch('/verify',{method:'POST',body:f})).json();const el=$('#verdict');
  if(!v.ok){el.className='';say(v.message,true);return}
  el.className=v.verified?'same':'diff';$('#vtext').textContent=v.verified?'SAME PERSON':'DIFFERENT PERSONS';
  $('#vdetails').innerHTML=`distance <b>${v.distance}</b> vs threshold <b>${v.threshold}</b> (${v.metric}) · model ${v.model} · detector ${v.detector} · ${v.seconds}s<br>
    <span class="muted">face areas: photo 1 [${v.area1.join(', ')}] · photo 2 [${v.area2.join(', ')}]</span>`;
  const col=v.verified?'#34d399':'#ef4444';if(v.area1[0]!=null)box(1,v.area1,col);if(v.area2[0]!=null)box(2,v.area2,col);
  say(v.verified?'The two photos belong to the same person.':'The two photos belong to different persons.')}
function setBusy(b){$('#shot1').disabled=b;$('#shot2').disabled=b||!shots[1];$('#reset').disabled=b}
$('#start').onclick=async()=>{try{video.srcObject=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});
  $('#start').disabled=true;setBusy(false)}catch(e){say('camera: '+e.message,true)}};
$('#shot1').onclick=()=>capture(1);$('#shot2').onclick=()=>capture(2);
$('#reset').onclick=()=>{shots[1]=shots[2]=null;for(const n of [1,2]){const c=$('#c'+n);c.getContext('2d').clearRect(0,0,c.width,c.height);$('#d'+n).innerHTML=''}
  $('#verdict').className='';say('');setBusy(false)};
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    for stream in (sys.stdout, sys.stderr):  # DeepFace prints a glyph cp1252 cannot encode
        with contextlib.suppress(ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Load everything at boot so the first click does not pay for model
    # downloads: the four attribute models are fetched once into ~/.deepface.
    from deepface import DeepFace
    from deepface.modules import modeling

    print("loading face models (first run downloads the age/gender/race/emotion weights) ...", flush=True)
    DeepFace.build_model(MODEL)
    for task, name in (("facial_attribute", "Age"), ("facial_attribute", "Gender"),
                       ("facial_attribute", "Race"), ("facial_attribute", "Emotion")):
        try:
            modeling.build_model(task=task, model_name=name)
        except Exception as exc:  # noqa: BLE001 - a missing attribute model only degrades /analyze
            print(f"  {name} model unavailable: {exc}", flush=True)

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"open http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
