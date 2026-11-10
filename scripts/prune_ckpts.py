"""Keep every 500th checkpoint + the 2 newest model files; keep only the 2 newest optimizer files. (macOS `head -n -2` is unsupported, which silently disabled the shell pruner.)
  python scripts/prune_ckpts.py DIR [--loop SECONDS]"""
import glob, os, sys, time
def prune(d):
    ms = sorted(glob.glob(os.path.join(d, "model_*.pt")))
    for f in ms[:-2]:
        s = int(os.path.basename(f)[6:12])
        if s % 500:
            for g in (f, os.path.join(d, f"meta_{s:06d}.json")):
                if os.path.exists(g): os.remove(g)
    for f in sorted(glob.glob(os.path.join(d, "optim_*_rank0.pt")))[:-2]: os.remove(f)
if __name__ == "__main__":
    d = sys.argv[1]; loop = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--loop" else 0
    prune(d)
    while loop: time.sleep(loop); prune(d)
