import os
import re
import threading
import base64
from functools import lru_cache
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
import cv2
import requests
from PIL import Image, PngImagePlugin
import piexif
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox, ttk, scrolledtext
from io import BytesIO
import numpy as np

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_PROMPT = "Output exactly 10 keywords describing the image. Comma separated. Stop after 10."
MAX_CONCURRENT = 3

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.webp', '.bmp')
VIDEO_EXTENSIONS = ('.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.ts')
ALL_EXTENSIONS   = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS

FILENAME_CLEAN   = re.compile(r'^(?:photograph|photo|image|picture|shot|view|讽刺)(?:\s+of)?(?:\s+(?:a|an|the))?\s+|^(?:a|an|the)\s+', re.IGNORECASE)
FILENAME_INVALID = re.compile(r'[<>:"/\\|?*\.;!\']')
FILENAME_SPACES  = re.compile(r'\s+')

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_CONCURRENT * 2,
                                        pool_maxsize=MAX_CONCURRENT * 2,
                                        max_retries=0)
session.mount('http://', adapter)
session.mount('https://', adapter)

# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_image(source, max_size=1500):
    img = Image.open(source) if isinstance(source, (str, os.PathLike)) else source
    if max(img.size) > max_size:
        r = max_size / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.Resampling.BICUBIC)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=75)
    if isinstance(source, str):
        img.close()
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def is_good_frame(frame, blur_threshold=50.0, dark_threshold=20):
    # Check darkness — mean pixel value across grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < dark_threshold:
        return False
    # Check blur — laplacian variance
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:
        return False
    return True

def extract_video_frames(path, num_frames=4, max_side=720):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, f"Cannot open video: {os.path.basename(path)}"
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None, None, "Video has no frames"
    duration = frame_count / fps

    # Sample a larger candidate pool, filter bad frames, pick evenly spaced survivors
    candidates = np.linspace(0, duration, num_frames * 4, endpoint=False)
    good_frames, good_ts = [], []
    for t in candidates:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        if not is_good_frame(frame):
            continue
        h, w  = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        good_frames.append(frame)
        good_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.release()

    if not good_frames:
        return None, None, "No usable frames found"

    # Evenly subsample down to num_frames if we have more
    if len(good_frames) > num_frames:
        indices = np.linspace(0, len(good_frames) - 1, num_frames, dtype=int)
        good_frames = [good_frames[i] for i in indices]
        good_ts     = [good_ts[i]     for i in indices]

    return good_frames, {"duration": duration, "timestamps": good_ts, "fps": fps}, None

def format_timestamp(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    return f"{h}:{m:02d}:{int(s%60):02d}" if h else f"{m}:{int(s%60):02d}"

# ── PreparedImage ─────────────────────────────────────────────────────────────

class PreparedImage:
    __slots__ = ('file_path', 'base64_data', 'video_info')
    def __init__(self, path, data, video_info=None):
        self.file_path  = path
        self.base64_data = data
        self.video_info  = video_info
    def cleanup(self):
        self.base64_data = None

# ── Preprocessing pipeline ───────────────────────────────────────────────────

def preprocessing_worker(in_q, out_q):
    while True:
        path = in_q.get()
        if path is None:
            break
        try:
            if path.lower().endswith(IMAGE_EXTENSIONS):
                out_q.put(PreparedImage(path, encode_image(path, max_size=1024)))
            elif path.lower().endswith(VIDEO_EXTENSIONS):
                frames, info, err = extract_video_frames(path)
                if err:
                    out_q.put(PreparedImage(path, None))
                else:
                    b64_frames = []
                    for f in frames:
                        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            b64_frames.append(f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}")
                    out_q.put(PreparedImage(path, b64_frames, info))
            else:
                out_q.put(PreparedImage(path, None))
        except Exception as e:
            print(f"Preprocessing error {path}: {e}")
            out_q.put(PreparedImage(path, None))
        finally:
            in_q.task_done()

# ── Caption generation ────────────────────────────────────────────────────────

def generate_caption(prepared, prompt, api_url, max_tokens, temperature, top_p, max_retries=2):
    if not prepared.base64_data:
        return None, "Failed to prepare image"

    is_video = isinstance(prepared.base64_data, list)
    if is_video:                         # ← no attempt check here
        info   = prepared.video_info
        frames = [f"Frame {i+1} ({format_timestamp(t)})" for i, t in enumerate(info['timestamps'])]
        system = "You are a video summarizer. Always respond with a single short sentence. Never use bullet points or frame-by-frame descriptions."
        text   = (f"These {len(prepared.base64_data)} frames span {format_timestamp(info['duration'])} of video.\n"
                  f"Timestamps: {', '.join(frames)}\n\n"
                  f"Write one concise sentence summarizing what happens in this video."
                  f"Focus on the main subject and action. DON'T describe frames individually. "
                  f"Use active voice. State actions directly and confidently. ")
        content = [{"type": "text", "text": text}] + \
                  [{"type": "image_url", "image_url": {"url": d, "detail": "low"}}
                   for d in prepared.base64_data]
        effective_max_tokens = max(max_tokens, 100)
    else:
        system  = "/no_think"
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": prepared.base64_data, "detail": "auto"}}]
        effective_max_tokens = max_tokens

    payload = {
        "model": "local-model",
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": effective_max_tokens,  # ← use override
        "repeat_penalty": 1.1,
        "stream": False
    }

    for attempt in range(max_retries + 1):
        resp = None
        try:
            if attempt == 0 and is_video:   # ← sleep lives here only
                time.sleep(1.0)
            elif attempt:
                time.sleep(5 if is_video else 2)
            resp = session.post(api_url, json=payload, timeout=(10, 120))
            resp.raise_for_status()
            result = resp.json()
            if result.get('choices'):
                raw = result['choices'][0]['message']['content'].strip()
                # Strip trailing truncated keyword after last comma
                if "," in raw:
                    last = raw.rfind(",")
                    after = raw[last + 1:].strip()
                    if " " not in after and len(after) < 4:
                        raw = raw[:last].strip()
                return raw, None
            return None, "No response from API"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == max_retries:
                return None, f"Failed after {max_retries+1} attempts: {e}"
        except Exception as e:
            return None, str(e)
        finally:
            if resp:
                resp.close()

# ── File operations ───────────────────────────────────────────────────────────

def clean_caption_for_filename(text, max_length):
    text = FILENAME_CLEAN.sub('', str(text)).strip()
    if not text:
        return "Untitled"
    text = text[0].upper() + text[1:]
    text = FILENAME_INVALID.sub('', text)
    text = FILENAME_SPACES.sub(' ', text).strip()
    if len(text) > max_length:
        cut = text.rfind(' ', max_length - 20, max_length)
        text = text[:cut if cut != -1 else max_length]
    return text or "Untitled"

def apply_rename(path, caption, mode):
    if mode == "none" or not caption:
        return path, None
    dirname  = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    clean    = clean_caption_for_filename(caption, 200)
    new_stem = f"{stem.strip()} - {clean}" if mode == "append" else clean
    max_len  = 250 - len(dirname) - len(ext) - 5
    if len(new_stem) > max_len:
        new_stem = new_stem[:max_len].rsplit(' ', 1)[0] + "..."
    new_path = os.path.join(dirname, f"{new_stem}{ext}")
    if os.path.exists(new_path) and os.path.normpath(new_path) != os.path.normpath(path):
        for i in range(1, 1000):
            sfx  = f" ({i})"
            base = new_stem if len(new_stem) + len(sfx) <= max_len else new_stem[:max_len - len(sfx) - 3] + "..."
            new_path = os.path.join(dirname, f"{base}{sfx}{ext}")
            if not os.path.exists(new_path):
                break
    try:
        os.rename(path, new_path)
        return new_path, None
    except OSError as e:
        return path, f"{os.path.basename(path)}: {e}"

def apply_metadata(path, caption):
    if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return
    try:
        stats = os.stat(path)
        with Image.open(path) as img:
            if img.format == 'PNG':
                info = PngImagePlugin.PngInfo()
                for k, v in img.info.items():
                    if isinstance(k, str) and isinstance(v, str):
                        info.add_text(k, v)
                info.add_text("Comment", caption)
                info.add_text("Description", caption)
                img.save(path, pnginfo=info)
            elif img.format in ('JPEG', 'JPEG2000'):
                exif = {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}, 'thumbnail': None}
                try:
                    if 'exif' in img.info:
                        exif = piexif.load(img.info['exif'])
                except Exception:
                    pass
                exif['0th'][piexif.ImageIFD.ImageDescription] = caption.encode('utf-8')
                try:
                    img.save(path, exif=piexif.dump(exif), quality="keep")
                except Exception:
                    img.save(path, quality="keep")
        os.utime(path, (stats.st_atime, stats.st_mtime))
    except Exception:
        pass

# ── Drop path parsing ─────────────────────────────────────────────────────────

def parse_drop_paths(event):
    try:
        return list(event.widget.tk.splitlist(event.data))
    except Exception:
        pass
    data = (event.data or "").strip()
    if '{' in data:
        return re.findall(r'\{([^}]+)\}', data)
    parts, paths, cur = data.split(' '), [], ''
    for p in parts:
        if re.match(r'^[A-Za-z]:[\\/]', p):
            if cur:
                paths.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        paths.append(cur)
    return [p.strip() for p in paths if p.strip()]

# ── Processing ────────────────────────────────────────────────────────────────

def process_files(file_paths, prompt, rename_mode, metadata_var, api_url,
                  message_label, result_label, progress_bar,
                  token_var, temperature_var, top_p_var):

    def gui(fn): root.after(0, fn)
    def set_result(text):
        def _():
            result_label.config(state='normal')
            result_label.delete('1.0', tk.END)
            result_label.insert('1.0', text)
            result_label.config(state='disabled')
        gui(_)

    progress_bar['maximum'] = len(file_paths)
    progress_bar['value']   = 0
    total = len(file_paths)

    def worker():
        rename_errors = []
        in_q  = Queue()
        out_q = Queue(maxsize=MAX_CONCURRENT * 3)
        has_video = any(fp.lower().endswith(VIDEO_EXTENSIONS) for fp in file_paths)
        effective_concurrent = 1 if has_video else MAX_CONCURRENT

        prep_threads = [threading.Thread(target=preprocessing_worker, args=(in_q, out_q), daemon=True)
                        for _ in range(MAX_CONCURRENT)]
        for t in prep_threads: t.start()
        for fp in file_paths:  in_q.put(fp)
        for _  in prep_threads: in_q.put(None)

        completed = 0
        with ThreadPoolExecutor(max_workers=effective_concurrent) as ex:
            pending, submitted = {}, 0
            for _ in range(min(effective_concurrent, total)):
                p = out_q.get()
                fut = ex.submit(generate_caption, p, prompt, api_url,
                                token_var.get(), temperature_var.get(), top_p_var.get())
                pending[fut] = p
                submitted += 1

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    prep     = pending.pop(fut)
                    fp       = prep.file_path
                    caption, err = fut.result()
                    prep.cleanup()
                    completed += 1

                    if submitted < total:
                        p = out_q.get()
                        nf = ex.submit(generate_caption, p, prompt, api_url,
                                       token_var.get(), temperature_var.get(), top_p_var.get())
                        pending[nf] = p
                        submitted += 1

                    gui(lambda c=completed, n=os.path.basename(fp):
                        message_label.config(text=f"Processing {c}/{total}: {n}"))
                    gui(lambda v=completed: progress_bar.config(value=v))

                    if err or not caption:
                        set_result(f"[{completed}] Failed: {os.path.basename(fp)}\n{err}")
                        continue

                    set_result(f"[{completed}] {caption}")

                    new_path = fp
                    mode = rename_mode.get()
                    if mode in ("append", "replace"):
                        new_path, rerr = apply_rename(fp, caption, mode)
                        if rerr:
                            rename_errors.append(rerr)

                    if metadata_var.get() and os.path.exists(new_path):
                        apply_metadata(new_path, caption)

        for t in prep_threads: t.join()

        if rename_errors:
            msg = "\n".join(rename_errors)
            gui(lambda m=msg: messagebox.showwarning("Rename Errors", m))

        summary = f"✓ Complete: {total} files" + (f" ({len(rename_errors)} rename errors)" if rename_errors else "")
        gui(lambda s=summary: message_label.config(text=s))
        gui(lambda: progress_bar.config(value=0))

    threading.Thread(target=worker, daemon=True).start()

def on_drop(event):
    seen, file_paths = set(), []
    for p in parse_drop_paths(event):
        p = os.path.normpath(re.sub(r'^file:/+', '', p.strip('"')))
        if p in seen or not os.path.isfile(p) or not p.lower().endswith(ALL_EXTENSIONS):
            continue
        seen.add(p)
        file_paths.append(p)

    if not file_paths:
        messagebox.showwarning("No valid files", "No supported image/video files detected.")
        return
    process_files(file_paths,
                  prompt_text.get("1.0", tk.END).strip() or DEFAULT_PROMPT,
                  rename_mode, metadata_var,
                  api_url_entry.get().strip() or API_URL,
                  message_label, result_label, progress_bar,
                  token_var, temperature_var, top_p_var)

# ── GUI ───────────────────────────────────────────────────────────────────────
root = TkinterDnD.Tk()
root.title("LM Studio Captioner")
root.geometry("420x700")
frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=5, pady=4)

# API
api_frame = tk.LabelFrame(frame, text="API Configuration")
api_frame.pack(fill="x", pady=(0, 2))
tk.Label(api_frame, text="API URL:").pack(anchor="w")
api_url_entry = tk.Entry(api_frame)
api_url_entry.insert(0, API_URL)
api_url_entry.pack(fill="x")

# Options
opt_frame = tk.LabelFrame(frame, text="Options & Generation Settings")
opt_frame.pack(fill="x", pady=2)

rename_mode  = tk.StringVar(value="append")
metadata_var = tk.BooleanVar(value=False)

rename_row = tk.Frame(opt_frame)
rename_row.pack(fill="x", pady=(2, 0))
tk.Label(rename_row, text="Rename:").pack(side="left")
for txt, val in [("Append", "append"), ("Replace", "replace"), ("None", "none")]:
    tk.Radiobutton(rename_row, text=txt, variable=rename_mode, value=val).pack(side="left")
tk.Checkbutton(opt_frame, text="Write metadata (PNG/JPG EXIF)", variable=metadata_var).pack(anchor="w")

# Sliders
token_var       = tk.IntVar(value=35)
temperature_var = tk.DoubleVar(value=0.4)
top_p_var       = tk.DoubleVar(value=0.95)

for label, var, lo, hi, res in [
    ("Tokens", token_var,       8,   128, 1),
    ("Temp",   temperature_var, 0.0, 1.1, 0.05),
    ("Top-P",  top_p_var,       0.1, 1.0, 0.05),
]:
    row = tk.Frame(opt_frame)
    row.pack(fill="x")
    tk.Label(row, text=label, width=7, anchor="w").pack(side="left")
    val_lbl = tk.Label(row, width=5, anchor="e", relief="sunken")
    val_lbl.pack(side="right", padx=(2, 4))
    var.trace_add("write", lambda *_, v=var, l=val_lbl: l.config(text=str(v.get())))
    val_lbl.config(text=str(var.get()))
    tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
             orient="horizontal", showvalue=0).pack(side="left", fill="x", expand=True)

# Drop zone
drop_label = tk.Label(frame, text="Drag & drop files here", height=3, bg="lightgrey")
drop_label.pack(fill="x", pady=2)
drop_label.drop_target_register(DND_FILES)
drop_label.dnd_bind('<<Drop>>', on_drop)

progress_bar = ttk.Progressbar(frame, orient="horizontal")
progress_bar.pack(fill="x", pady=2)

# Prompt
prompt_frame = tk.LabelFrame(frame, text="Prompt")
prompt_frame.pack(fill="x", pady=2)
prompt_text = scrolledtext.ScrolledText(prompt_frame, height=4)
prompt_text.insert("1.0", DEFAULT_PROMPT)
prompt_text.pack(fill="x")

# Result
result_frame = tk.LabelFrame(frame, text="Last Generated Caption")
result_frame.pack(fill="both", expand=True, pady=2)
result_label = scrolledtext.ScrolledText(result_frame, state="disabled")
result_label.pack(fill="both", expand=True)

message_label = tk.Label(frame, text="Ready")
message_label.pack(fill="x")

root.mainloop()
