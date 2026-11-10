"""Keynote-style demo video of the 500M model, narrated as a product manager. Every number comes from evaluation/results (see reports/progress.md);
the Q&A beats are real model outputs selected from evaluation/results/demo500_selection.json and labelled as selected. Pillow + macOS `say` + ffmpeg.
  /path/to/venv_with_pillow/bin/python scripts/make_demo_video.py"""
import os, subprocess, textwrap
from PIL import Image, ImageDraw, ImageFont
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); W, H = 1920, 1080; OUT = os.path.join(R, "video"); T = os.path.join(OUT, "tmp"); os.makedirs(T, exist_ok=True)
F = "/System/Library/Fonts/SFNS.ttf"
def font(sz, w=None):
    f = ImageFont.truetype(F, sz)
    try: f.set_variation_by_axes([w or 400])
    except Exception: pass
    return f
BG, FG, MUT, ACC, WARN = (0, 0, 0), (245, 245, 247), (134, 134, 139), (41, 151, 255), (255, 159, 10)
def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]
def frame(path, head=None, sub=None, q=None, a=None, tag=None, head_color=FG):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if q is not None:                                    # Q&A beat
        fq, fa = font(54, 500), font(96, 700); y = 250
        if tag: d.text((160, 120), tag, font=font(30, 500), fill=MUT)
        for l in wrap(d, q, fq, W - 320): d.text((160, y), l, font=fq, fill=MUT); y += 74
        y += 50
        if a is not None:
            d.text((160, y), "Model", font=font(30, 600), fill=ACC); y += 56
            for l in wrap(d, a, fa, W - 320): d.text((160, y), l, font=fa, fill=FG); y += 118
    else:
        fh, fs = font(120, 700), font(46, 400); lines = wrap(d, head, fh, W - 360); y = (H - len(lines) * 140 - (130 if sub else 0)) // 2
        for l in lines: d.text(((W - d.textlength(l, font=fh)) / 2, y), l, font=fh, fill=head_color); y += 140
        if sub:
            y += 20
            for l in wrap(d, sub, fs, W - 480): d.text(((W - d.textlength(l, font=fs)) / 2, y), l, font=fs, fill=MUT); y += 64
    im.save(path)
BEATS = [  # (frame kwargs, narration)
 (dict(head="NanoChat 500M.", sub="A language model, built from scratch."), "Meet NanoChat five hundred M."),
 (dict(head="495 million parameters.", sub="Trained on a single Mac mini."), "A language model with four hundred ninety-five million parameters. Trained from scratch, on one Mac mini."),
 (dict(head="Trained on science and open knowledge.", sub="649 million tokens. Research abstracts, Wikipedia, textbooks, books, and Q and A."), "It learned from science papers, Wikipedia, textbooks and books. Six hundred forty-nine million tokens, the small pieces of text a model reads."),
 (dict(head="Give it a passage. Ask a question.", sub=None), "Here is what it does well. Give it a passage, and ask a question."),
 (dict(q="Where is Energiprojekt AB based?", a="Sweden", tag="Passage-based question · selected correct answer"), "Where is Energiprojekt based? Sweden."),
 (dict(q="What function is related to prime numbers?", a="the zeta function", tag="Passage-based question · selected correct answer"), "What function is related to prime numbers? The zeta function."),
 (dict(q="The lancelet lacks what feature that distinguishes vertebrates?", a="backbone.", tag="Science question, no passage · selected correct answer"), "Even without a passage, sometimes it knows. What does the lancelet lack? A backbone."),
 (dict(q="What is a reptile's skin covered in to protect it from drying out?", a="scales.", tag="Science question, no passage · selected correct answer"), "What covers a reptile's skin? Scales."),
 (dict(head="Now, the honest part.", sub="These were selected. Here is the real scoreboard.", head_color=WARN), "Now the honest part. Those answers were selected. Here is the real scoreboard."),
 (dict(head="21%", sub="of passage-based questions answered correctly. 42 of 200 held-out questions."), "With a passage, it answers twenty-one percent of held-out questions correctly."),
 (dict(head="5%", sub="of science questions without a passage. 20 of 400."), "Without a passage, about five percent of science questions."),
 (dict(head="0.3%", sub="of trivia questions. 1 of 300. On multiple choice, near chance."), "Trivia, about one in three hundred. And on multiple choice, it is near chance. It is a small model, and we are not going to pretend otherwise."),
 (dict(head="What's next.", sub="1.2 billion tokens of Khan Academy, OpenStax, Stanford-style courses and MIT lectures. Training now."), "What is next? A new model, trained on one point two billion tokens of Khan Academy, OpenStax, course material and MIT lectures. It is training right now."),
 (dict(head="NanoChat 500M.", sub="Small. Local. Honest."), "NanoChat five hundred M. Small. Local. Honest."),
]
segs = []
for i, (kw, narr) in enumerate(BEATS):
    png, aif, mp4 = (os.path.join(T, f"{i:02d}.{e}") for e in ("png", "aiff", "mp4")); frame(png, **kw)
    subprocess.run(["say", "-v", "Samantha", "-r", "165", "-o", aif, narr], check=True)
    dur = float([l for l in subprocess.run(["afinfo", aif], capture_output=True, text=True).stdout.splitlines() if "estimated duration" in l][0].split()[2]) + 1.1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", png, "-i", aif, "-filter_complex",
                    f"[0:v]fade=t=in:st=0:d=0.5,fade=t=out:st={dur-0.5:.2f}:d=0.5,format=yuv420p[v];[1:a]adelay=500|500,apad[a]", "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "160k", mp4], check=True); segs.append(mp4)
lst = os.path.join(T, "list.txt"); open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
final = os.path.join(OUT, "nanochat_500m_demo.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
print(final, os.path.getsize(final) // 1024, "KB")
