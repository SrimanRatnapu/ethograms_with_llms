#!/usr/bin/env python3
import os, csv, json, time, random, base64, argparse, subprocess
from datetime import datetime
from pathlib import Path
from openai import OpenAI

LABELS = ["Resting","Locomotion","Feeding","Social","ObjectInteraction","OutOfFrame","Uncertain"]

SYSTEM_PROMPT = (
    "You are an expert animal behavior annotator trained in creating ethograms. "
    "You strictly follow provided behavior definitions and do not infer unobservable internal states. "
    'If behavior cannot be determined confidently, label it as "Uncertain".'
)

USER_PROMPT = """You are given a short video subclip represented by a few extracted frames of a captive animal.

Task:
Assign ONE dominant behavior label for the focal animal for this 10-second subclip.

Behavior labels (choose exactly one):
- Resting: Lying or sitting with no locomotion.
- Locomotion: Walking, climbing, or swimming.
- Feeding: Eating or manipulating food.
- Social: Direct interaction with another animal (physical contact, grooming, directed gestures).
- ObjectInteraction: Manipulating toys or enclosure objects.
- OutOfFrame: Focal animal not visible for most of the subclip.
- Uncertain: Behavior cannot be confidently determined.

Rules:
- Use only visible evidence in the frames.
- Do NOT infer emotional or internal states.
- If multiple behaviors occur, choose the one occupying the majority of the subclip.
- If you cannot confidently assign a label, use Uncertain.

Return STRICT JSON ONLY (no markdown, no extra text) with this schema:
{
  "behavior_label": "<one of: Resting | Locomotion | Feeding | Social | ObjectInteraction | OutOfFrame | Uncertain>",
  "confidence": <number between 0 and 1>,
  "notes": "<very short reason, <= 15 words>"
}
"""

def extract_frames(mp4_path: Path, out_dir: Path, times):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames=[]
    for t in times:
        out_file = out_dir / f"frame_{t:02d}.jpg"
        subprocess.run(["ffmpeg","-y","-ss",str(t),"-i",str(mp4_path),"-frames:v","1","-q:v","2",str(out_file)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames.append(out_file)
    return frames

def img_to_data_url(p: Path) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ["429","rate limit","timeout","timed out","overloaded","503","502","504","connection"])

def call_with_retries(client, model, content_parts, max_retries=8):
    attempt=0
    while True:
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role":"system","content":[{"type":"input_text","text":SYSTEM_PROMPT}]},
                    {"role":"user","content":content_parts},
                ],
            )
            return resp.output_text
        except Exception as e:
            attempt += 1
            if attempt > max_retries or not is_transient(e):
                raise
            delay = min(60, (2 ** (attempt-1)) * 1.5)
            delay += random.uniform(0, 0.8*delay)
            time.sleep(delay)

def load_done(out_csv: Path):
    if not out_csv.exists(): return set()
    done=set()
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            if row.get("filename"):
                done.add(row["filename"])
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default="clips")
    ap.add_argument("--out_csv", default="labels.csv")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4o"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    clips_dir = Path(args.clips_dir)
    mp4s = sorted(clips_dir.glob("*.mp4"))
    out_csv = Path(args.out_csv)
    done = load_done(out_csv) if args.resume else set()

    if args.frames <= 1:
        times=[5]
    elif args.frames == 3:
        times=[1,5,9]
    else:
        step = 10/args.frames
        times = sorted({max(1,min(9,int(round(i*step)))) for i in range(args.frames)})
        if not times: times=[5]

    client = OpenAI()
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename","behavior_label","confidence","notes","model","frames_used","raw_json","created_at"])
        if write_header: w.writeheader()

        for mp4 in mp4s:
            if mp4.name in done:
                continue

            tmp = Path("tmp_frames")/mp4.stem
            frames = extract_frames(mp4, tmp, times)
            content = [{"type":"input_text","text":USER_PROMPT}] + [{"type":"input_image","image_url":img_to_data_url(p)} for p in frames]

            raw = call_with_retries(client, args.model, content)
            try:
                out = json.loads(raw.strip())
            except Exception:
                out = {"behavior_label":"Uncertain","confidence":0.0,"notes":"PARSE_FAIL"}

            label = out.get("behavior_label","Uncertain")
            conf = out.get("confidence",0.0)
            notes = out.get("notes","")
            if label not in LABELS:
                label="Uncertain"; conf=0.0; notes=(notes+" INVALID_LABEL").strip()

            w.writerow({
                "filename": mp4.name,
                "behavior_label": label,
                "confidence": conf,
                "notes": notes,
                "model": args.model,
                "frames_used": ";".join([p.name for p in frames]),
                "raw_json": json.dumps(out)[:2000],
                "created_at": datetime.utcnow().isoformat()+"Z",
            })
            f.flush()

            for p in frames:
                try: p.unlink()
                except: pass
            try: tmp.rmdir()
            except: pass

            time.sleep(0.4 + random.uniform(0,0.8))

if __name__ == "__main__":
    main()
