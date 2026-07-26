"""Test battery for beam_line_tree: fonts x lines, reports per-case and
aggregate accuracy (exact, case-insensitive, char error rate)."""
import os
import subprocess
import sys
import difflib

LINES = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "Hello World this is a test",
    "The train arrives at 7 pm",
    "watch out, it bites!",
    "Price is $42.50 today",
    "seventeen paper towers stood",
    "a gentle breeze moves slowly",
]

FONTS = [
    "fonts/Arial.ttf",
    "fonts/Times New Roman.ttf",
    "fonts/Courier New.ttf",
    "fonts/Georgia.ttf",
    "fonts/Verdana.ttf",
    "fonts/Comic Sans MS.ttf",
]


def cer(a, b):
    sm = difflib.SequenceMatcher(None, a, b)
    return 1.0 - sm.ratio()


def main():
    results = []
    total = 0
    for font in FONTS:
        for line in LINES:
            total += 1
            cmd = [
                ".venv/bin/python", "beam_line_tree.py",
                "--line", line, "--font", font,
                "--conf-threshold", "0.1",
                "--max-expansions", "600000",
                "--two-pass",
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=900,
                    env=dict(os.environ),
                ).stdout
            except subprocess.TimeoutExpired:
                results.append((font, line, None, "TIMEOUT"))
                print(f"[{total}] {font.split('/')[-1]:<24} {line!r}: TIMEOUT",
                      flush=True)
                continue
            top1 = None
            for l in out.splitlines():
                if l.startswith("Decoded (top-1):"):
                    top1 = l.split(":", 1)[1].strip().strip("'\"")
            if top1 is None:
                results.append((font, line, None, "FAIL"))
                print(f"[{total}] {font.split('/')[-1]:<24} {line!r}: NO OUTPUT",
                      flush=True)
                continue
            exact = top1 == line
            ci = top1.lower() == line.lower()
            e = cer(top1.lower(), line.lower())
            results.append((font, line, top1, (exact, ci, e)))
            tag = "EXACT" if exact else ("CASE" if ci else f"cer={e:.2f}")
            print(f"[{total}] {font.split('/')[-1]:<24} {line!r} -> {top1!r} [{tag}]",
                  flush=True)

    done = [r for r in results if isinstance(r[3], tuple)]
    n = len(done)
    if n:
        ex = sum(1 for r in done if r[3][0])
        ci = sum(1 for r in done if r[3][1])
        avg_cer = sum(r[3][2] for r in done) / n
        print(f"\n=== {n}/{total} completed ===")
        print(f"exact:            {ex}/{n} ({100*ex/n:.0f}%)")
        print(f"case-insensitive: {ci}/{n} ({100*ci/n:.0f}%)")
        print(f"mean CER (ci):    {avg_cer:.3f}")


if __name__ == "__main__":
    main()
