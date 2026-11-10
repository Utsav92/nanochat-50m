"""Product-demo video (UI walkthrough) of the 500M chat model. All UI frames are RENDERED with Pillow from recorded real data (not screen captures):
  evaluation/results/demo500_ui_data.json (real prompts/outputs/inference stats from the preview server), sci500_sft.json / q500_sft.json / sci500m_base.json (evals), logs/pre500.log (loss curve).
  Voiceover: macOS `say`. Run:  /path/to/venv_with_pillow/bin/python scripts/make_ui_demo_video.py   -> video/nanochat_500m_ui_demo.mp4"""
import json, os, re, subprocess
from PIL import Image, ImageDraw, ImageFont
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); W, H = 1920, 1080; OUT = os.path.join(R, "video"); T = os.path.join(OUT, "tmp2"); os.makedirs(T, exist_ok=True)
D = json.load(open(f"{R}/evaluation/results/demo500_ui_data.json")); SC = json.load(open(f"{R}/evaluation/results/sci500_sft.json")); Q = json.load(open(f"{R}/evaluation/results/q500_sft.json"))
BG, PANEL, PANEL2, LINE = (0, 0, 0), (28, 28, 30), (44, 44, 46), (58, 58, 60); FG, MUT, ACC, GOOD, BAD, WARN = (245, 245, 247), (152, 152, 157), (10, 132, 255), (48, 209, 88), (255, 69, 58), (255, 159, 10)
def font(sz, w=400):
    f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", sz); f.set_variation_by_axes([100, max(17, min(96, sz)), 400, w]); return f
def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]
def rr(d, box, r, fill, outline=None): d.rounded_rectangle(box, r, fill=fill, outline=outline, width=2 if outline else 0)
def pill(d, x, y, text, color=MUT, f=None):
    f = f or font(26, 500); w = d.textlength(text, font=f) + 36; rr(d, (x, y, x + w, y + 48), 24, PANEL2); d.text((x + 18, y + 8), text, font=f, fill=color); return x + w + 12
def title_frame(head, sub=None, color=FG):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); fh, fs = font(116, 700), font(44); lines = wrap(d, head, fh, W - 360); y = (H - len(lines) * 136 - (120 if sub else 0)) // 2
    for l in lines: d.text(((W - d.textlength(l, font=fh)) / 2, y), l, font=fh, fill=color); y += 136
    for l in (wrap(d, sub, fs, W - 480) if sub else []): y += 12; d.text(((W - d.textlength(l, font=fs)) / 2, y), l, font=fs, fill=MUT); y += 60
    return im
def ui_frame(model="sft/sci500m_b", messages=(), rag=False, caption=None):
    """Chat window: sidebar (checkpoints), header (model facts), conversation, composer. messages: list of dict(role, text, chips, verdict, sources, passage)."""
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); X0, Y0, X1, Y1 = 90, 60, W - 90, H - 110; rr(d, (X0, Y0, X1, Y1), 28, PANEL, LINE)
    d.line((X0 + 380, Y0, X0 + 380, Y1), fill=LINE, width=2); d.text((X0 + 36, Y0 + 30), "NanoChat", font=font(40, 700), fill=FG)
    y = Y0 + 110; d.text((X0 + 36, y), "CHECKPOINTS", font=font(22, 600), fill=MUT); y += 48
    for k, label in (("base/sci500m", "Base · sci500m"), ("sft/sci500m_b", "Chat · sci500m_b (SFT)"), ("rl/sci500m_b", "RL · sci500m_b (step 25)")):
        sel = k == model
        if sel: rr(d, (X0 + 20, y - 6, X0 + 360, y + 52), 14, PANEL2)
        d.text((X0 + 36, y + 4), label, font=font(27, 600 if sel else 400), fill=FG if sel else MUT); y += 70
    y += 20; d.text((X0 + 36, y), "RETRIEVAL (RAG)", font=font(22, 600), fill=MUT); y += 48; rr(d, (X0 + 36, y, X0 + 116, y + 44), 22, ACC if rag else PANEL2)
    d.ellipse((X0 + (84 if rag else 42), y + 5, X0 + (110 if rag else 68), y + 39), fill=FG); d.text((X0 + 134, y + 6), "Wikipedia passages" if rag else "Off", font=font(25), fill=FG if rag else MUT)
    cx = X0 + 380; d.text((cx + 40, Y0 + 28), "494.9M parameters", font=font(30, 600), fill=FG); d.text((cx + 40, Y0 + 70), "Runs on this Mac · answers greedy (temperature 0)", font=font(24), fill=MUT)
    d.line((cx, Y0 + 120, X1, Y0 + 120), fill=LINE, width=2); y = Y0 + 150; fm, fs = font(34), font(23)
    for m in messages:
        maxw = X1 - cx - 220
        if m["role"] == "user":
            lines = wrap(d, m["text"], fm, maxw); w = max(d.textlength(l, font=fm) for l in lines) + 56; hh = len(lines) * 46 + 36; rr(d, (X1 - 40 - w, y, X1 - 40, y + hh), 26, ACC)
            for i, l in enumerate(lines): d.text((X1 - 40 - w + 28, y + 18 + i * 46), l, font=fm, fill=FG)
            y += hh + 26
        else:
            if m.get("passage"):
                pl = wrap(d, m["passage"], font(22), X1 - cx - 160)[:4]; rr(d, (cx + 40, y, X1 - 40, y + len(pl) * 30 + 56), 14, PANEL2); d.text((cx + 60, y + 10), "PASSAGE GIVEN TO THE MODEL", font=font(18, 600), fill=MUT)
                for i, l in enumerate(pl): d.text((cx + 60, y + 38 + i * 30), l + (" …" if i == 3 else ""), font=font(22), fill=MUT)
                y += len(pl) * 30 + 76
            lines = wrap(d, m["text"], fm, maxw); w = max(d.textlength(l, font=fm) for l in lines) + 56; hh = len(lines) * 46 + 36; rr(d, (cx + 40, y, cx + 40 + w, y + hh), 26, PANEL2)
            for i, l in enumerate(lines): d.text((cx + 68, y + 18 + i * 46), l, font=fm, fill=FG)
            y += hh + 14; x = cx + 40
            for c in m.get("chips", []): x = pill(d, x, y, c)
            if m.get("verdict"): ok, t = m["verdict"]; x = pill(d, x, y, t, GOOD if ok else BAD, font(26, 700))
            y += 64
            if m.get("sources"):
                x = cx + 40; d.text((x, y + 8), "Sources", font=font(24, 600), fill=MUT); x += 120
                for s in m["sources"]: x = pill(d, x, y, s[:34], MUT, font(23))
                y += 64
            y += 14
    rr(d, (cx + 40, Y1 - 100, X1 - 40, Y1 - 30), 35, PANEL2); d.text((cx + 76, Y1 - 82), "Ask a question…", font=font(30), fill=MUT)
    if caption: d.text((X0, Y1 + 28), caption, font=font(26), fill=MUT)
    return im
def chips(x): return [f"{x['tokens']} tokens", f"first token {x['first_token_ms']} ms", f"{x['tok_per_s']} tok/s"]
def bars_frame():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); d.text((160, 90), "Held-out evaluation", font=font(72, 700), fill=FG)
    d.text((160, 180), "Greedy decoding · 300 questions per test unless noted · 95% intervals are about ±5 points · single run", font=font(28), fill=MUT)
    sq, sc, tv = Q["squad"], SC, Q["trivia"]
    rows = [("SQuAD, passage given", sq["answerable_contains_gold"]["acc"], None, "reference answer found in reply"),
            ("SciQ, no passage", sc["sciq"]["closed_book_contains_gold"]["acc"], None, "reference answer found in reply"),
            ("TriviaQA, no passage", tv["answer_contains_gold"]["acc"], tv["accidental_baseline"]["acc"], "accidental-match baseline 0.3%"),
            ("SciQ multiple choice", sc["sciq"]["mc_letter_acc"]["acc"], 0.25, "chance 25%"), ("MedQA multiple choice", sc["medqa"]["acc"]["acc"], 0.25, "chance 25%"),
            ("PubMedQA yes / no / maybe", sc["pubmedqa"]["acc"]["acc"], sc["pubmedqa"]["majority_class_baseline"]["acc"], "always answering “yes” = 54.7%"),
            ("HumanEval code (pass@1)", Q["humaneval"]["pass_at_1"]["acc"], None, "164 problems"), ("MBPP code (pass@1)", Q["mbpp"]["pass_at_1"]["acc"], None, "257 problems")]
    y = 270; bx, bw = 700, 760
    for name, v, base, note in rows:
        d.text((160, y + 4), name, font=font(32, 500), fill=FG); rr(d, (bx, y + 6, bx + bw, y + 46), 8, PANEL2); rr(d, (bx, y + 6, bx + max(8, int(bw * v)), y + 46), 8, ACC)
        if base is not None: x = bx + int(bw * base); d.line((x, y - 2, x, y + 54), fill=WARN, width=5)
        d.text((bx + bw + 24, y + 2), f"{100*v:.1f}%", font=font(36, 700), fill=FG); d.text((bx + bw + 190, y + 10), note, font=font(24), fill=MUT); y += 78
    d.text((160, y + 20), "Orange tick = chance or baseline. Bars at or below the tick mean no real skill.", font=font(26), fill=WARN); return im
def curve_frame():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); d.text((160, 90), "Trained from scratch", font=font(72, 700), fill=FG)
    pts = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"step (\d+)/09900 .*?loss: ([\d.]+)", open(f"{R}/logs/pre500.log").read())]; last = {}
    for s, l in pts: last[s] = l
    st = sorted(last); vals = [last[s] for s in st]; sm = []; a = None
    for v in vals: a = v if a is None else 0.98 * a + 0.02 * v; sm.append(a)
    x0, y0, x1, y1 = 200, 250, 1300, 880; rr(d, (x0 - 30, y0 - 30, x1 + 30, y1 + 40), 20, PANEL); lo, hi = 2.0, 7.0
    for g in (3, 4, 5, 6, 7): yy = y1 - (g - lo) / (hi - lo) * (y1 - y0); d.line((x0, yy, x1, yy), fill=LINE, width=1); d.text((x0 - 24 - 36, yy - 14), str(g), font=font(22), fill=MUT)
    xy = [(x0 + s / 9900 * (x1 - x0), y1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y1 - y0)) for s, v in zip(st, sm)]; d.line(xy, fill=ACC, width=5)
    d.text((x0, y1 + 8), "training step", font=font(22), fill=MUT); d.text((x0 - 150, y0 - 60), "loss", font=font(22), fill=MUT)
    for i, (big, small) in enumerate((("9,900", "steps"), ("649M", "tokens read"), ("2.3", "final training loss"), ("0.60", "validation bits per byte"))):
        d.text((1400, 280 + i * 150), big, font=font(72, 700), fill=FG); d.text((1400, 360 + i * 150), small, font=font(28), fill=MUT)
    return im
def inference_frame():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); d.text((160, 90), "Inference, on the Mac", font=font(72, 700), fill=FG)
    xs = [D["science"][0], D["science"][1], D["passage"], D["miss"], D["rag"][0]]; ft = [x["first_token_ms"] for x in xs]; ts = [x["tok_per_s"] for x in xs]
    for i, (big, small) in enumerate(((f"{min(ft)}–{max(ft)} ms", "to first token"), (f"{min(ts)}–{max(ts)}", "tokens per second"), ("494.9M", "parameters"), ("1,024", "tokens of context"))):
        col, row = i % 2, i // 2; d.text((160 + col * 800, 300 + row * 260), big, font=font(104, 700), fill=FG); d.text((160 + col * 800, 430 + row * 260), small, font=font(36), fill=MUT)
    d.text((160, 880), "Measured on five real requests while 500M training was running on the same Mac, so an idle machine should be faster.", font=font(28), fill=WARN); return im
S, P, M, RG, NR = D["science"], D["passage"], D["miss"], D["rag"], D["norag"]
BEATS = [
 (title_frame("NanoChat 500M.", "A language model, built from scratch."), "I'm excited to show you NanoChat five hundred M. A language model we trained from scratch on one Mac mini."),
 (ui_frame(caption="The preview app: pick a checkpoint, ask a question."), "This is the app. On the left, the checkpoints. On the right, you just ask."),
 (ui_frame(messages=[dict(role="user", text=S[0]["prompt"])]), "Let's ask a science question. What feature does the lancelet lack that vertebrates have?"),
 (ui_frame(messages=[dict(role="user", text=S[0]["prompt"]), dict(role="assistant", text=S[0]["answer"], chips=chips(S[0]), verdict=(True, "Correct · reference: backbone"))]), f"A backbone. Correct. {S[0]['tokens']} tokens, first token in {S[0]['first_token_ms']} milliseconds, all on this Mac."),
 (ui_frame(messages=[dict(role="user", text=S[1]["prompt"]), dict(role="assistant", text=S[1]["answer"], chips=chips(S[1]), verdict=(True, "Correct · reference: scales"))]), "Another one. What covers a reptile's skin? Scales. Also correct."),
 (ui_frame(messages=[dict(role="user", text=P["question"]), dict(role="assistant", passage=P["context"], text=P["answer"], chips=chips(P), verdict=(True, "Correct · reference: rationing"))]), "It is best when you give it a passage. Here it reads a paragraph about oil prices, and answers: by rationing. Correct."),
 (ui_frame(messages=[dict(role="user", text=M["prompt"]), dict(role="assistant", text=M["answer"], chips=chips(M), verdict=(False, "Wrong · reference: inherited disorder"))]), "But I want to be straight with you. Ask what kind of disorder Tay-Sachs is, and it just repeats the name. Wrong."),
 (ui_frame(rag=True, messages=[dict(role="user", text=RG[0]["prompt"]), dict(role="assistant", text=RG[0]["answer"], chips=chips(RG[0]), sources=sorted(set(RG[0]["sources"])), verdict=(False, "No answer · retrieval found a disambiguation page"))]), "We added retrieval from Wikipedia. For Pride and Prejudice it found only a disambiguation page, so the model says it doesn't know. That is honest, but not helpful yet."),
 (ui_frame(messages=[dict(role="user", text=NR[0]["prompt"]), dict(role="assistant", text=NR[0]["answer"], chips=chips(NR[0]), verdict=(False, "Wrong · the author is Jane Austen"))]), "Without retrieval, it makes up an answer. Also wrong. This is what a small model does when it doesn't know."),
 (bars_frame(), "Here are the results on held-out questions. With a passage, twenty-one percent. Without one, about five percent. On multiple choice, it sits near chance, and on yes-or-no medical questions it just says yes."),
 (inference_frame(), "On speed: first tokens in well under a second, on a laptop-class chip, measured while training was running in the background."),
 (curve_frame(), "And here is the training. Nine thousand nine hundred steps, six hundred forty-nine million tokens, from random weights."),
 (title_frame("Next: 1.2 billion tokens.", "Khan Academy, OpenStax, Stanford-style courses and MIT lectures. Training now. We will measure it the same way."), "What's next? A new model, trained on one point two billion tokens of textbooks, courses and lectures. It's training now. We'll measure it exactly the same way, and show you the results, good or bad."),
 (title_frame("NanoChat 500M.", "Small. Local. Honest."), "NanoChat five hundred M. Small. Local. Honest."),
]
segs = []
for i, (im, narr) in enumerate(BEATS):
    png, aif, mp4 = (os.path.join(T, f"{i:02d}.{e}") for e in ("png", "aiff", "mp4")); im.save(png)
    subprocess.run(["say", "-v", "Samantha", "-r", "170", "-o", aif, narr], check=True)
    dur = float([l for l in subprocess.run(["afinfo", aif], capture_output=True, text=True).stdout.splitlines() if "estimated duration" in l][0].split()[2]) + 1.1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", png, "-i", aif, "-filter_complex", f"[0:v]fade=t=in:st=0:d=0.4,fade=t=out:st={dur-0.4:.2f}:d=0.4,format=yuv420p[v];[1:a]adelay=450|450,apad[a]",
                    "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "160k", mp4], check=True); segs.append(mp4)
open(os.path.join(T, "list.txt"), "w").write("".join(f"file '{s}'\n" for s in segs)); final = os.path.join(OUT, "nanochat_500m_ui_demo.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(T, "list.txt"), "-c", "copy", final], check=True); print(final, os.path.getsize(final) // 1024, "KB")
