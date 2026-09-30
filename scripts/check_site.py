#!/usr/bin/env python
"""Load the live demo page in headless Chromium and check what it displays.

    python scripts/check_site.py                 build, serve, check (make site-check)
    python scripts/check_site.py --build _site   only assemble the site (pages.yml)

tests/test_demo_data.py proves the committed per-flow export agrees with the
committed metrics. It does not run the page's JavaScript, which re-implements
the reject rule by hand. This script does: it assembles the site from the
committed files exactly as pages.yml deploys it, serves it on localhost, and
drives the real page.

  - The knob: at every committed threshold, the number of flows answered must
    equal the committed curve's n_covered, coverage and reliability must match
    it, and every count in the confusion table and the per-family table must
    equal a recount done here in Python from the export. At threshold 0 the
    per-family table must equal per_family.json.
  - The held-out-family panel: for every family option and every threshold
    button, the three outcome shares, the known-traffic tile, the novelty lift,
    the shuffled-label control and the flow count must equal numbers computed
    here from zero_day_lofo.json.
  - The headline tiles, including the benign denominators at 0.99.
  - No JavaScript error, no load failure, and no horizontal page scroll at a
    390 px phone width or at 1280 px.

Needs `pip install playwright` and `python -m playwright install chromium`.
Uses only the standard library otherwise, so it does not import the package.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import re
import shutil
import sys
import tempfile
import threading
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
DATA_FILES = ["demo_flows.json", "zero_day_lofo.json", "metrics.json"]


def build_site(dest: Path) -> Path:
    """What pages.yml publishes: the page plus the committed data files."""
    (dest / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "site" / "index.html", dest / "index.html")
    for name in DATA_FILES:
        shutil.copy(ART / name, dest / "data" / name)
    return dest


def num(text: str) -> float:
    m = re.search(r"[-−]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        raise ValueError(f"no number in {text!r}")
    return float(m.group(0).replace(",", "").replace("−", "-"))


def nums(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"[\d,]*\.?\d+", text)]


def zpct(x: float) -> str:
    """The page's zpct(): 4 decimals first, then one decimal of a percent,
    rounding half up the way Math.round does."""
    r4 = math.floor(x * 1e4 + 0.5)
    return f"{math.floor(r4 / 10 + 0.5) / 10:.1f}%"


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def eq(self, what: str, got: Any, want: Any) -> None:
        self.checks += 1
        if got != want:
            self.failures.append(f"{what}: page shows {got!r}, expected {want!r}")

    def near(self, what: str, got: float, want: float, tol: float) -> None:
        self.checks += 1
        if abs(got - want) > tol + 1e-9:
            self.failures.append(f"{what}: page shows {got}, expected {want} (+/- {tol})")


def recount(demo: dict[str, Any], t: float) -> dict[str, Any]:
    """The reject rule, written again in Python: answered iff confidence >= t."""
    classes = demo["classes"]
    benign = classes.index("benign")
    c = Counter()
    fam = {k: [0, 0, 0, 0, 0] for k in range(len(classes))}
    for yk, pk, conf in zip(demo["true"], demo["pred"], demo["confidence"], strict=True):
        f = fam[yk]
        f[0] += 1
        is_b = yk == benign
        if conf < t:
            c["unk"] += 1
            f[4] += 1
            c["bu" if is_b else "au"] += 1
            continue
        if pk == yk:
            c["right"] += 1
            f[1] += 1
        else:
            c["wrong"] += 1
            f[3 if pk == benign else 2] += 1
        if is_b:
            c["bb" if pk == benign else "ba"] += 1
        else:
            c["ab" if pk == benign else "aa"] += 1
    return {"c": c, "fam": fam}


def set_knob(page, t: float) -> None:
    page.eval_on_selector(
        "#t", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", repr(t)
    )


def check_knob(page, ck: Checker, demo, metrics, per_family) -> None:
    classes = demo["classes"]
    for row in metrics["coverage_reliability_curve"]:
        t = row["threshold"]
        set_knob(page, t)
        ck.eq(f"slider value at {t}", float(page.input_value("#t")), t)
        r = recount(demo, t)
        c = r["c"]
        tx = lambda i: page.text_content(f"#{i}") or ""  # noqa: E731
        right, wrong, unk = num(tx("n-right")), num(tx("n-wrong")), num(tx("n-unknown"))
        ck.eq(f"t={t} flows answered vs committed n_covered", int(right + wrong), row["n_covered"])
        ck.eq(f"t={t} right/wrong/unknown", (right, wrong, unk), (c["right"], c["wrong"], c["unk"]))
        ck.near(f"t={t} coverage", num(tx("m-cov")), 100 * row["coverage"], 0.05)
        if row["reliability"] is not None:
            ck.near(f"t={t} reliability", num(tx("m-rel")), 100 * row["reliability"], 0.05)
        for cell, key in [("c-aa", "aa"), ("c-ab", "ab"), ("c-au", "au"),
                          ("c-ba", "ba"), ("c-bb", "bb"), ("c-bu", "bu")]:
            ck.eq(f"t={t} {cell}", int(num(tx(cell))), c[key])
        ck.eq(f"t={t} m-silent", int(num(tx("m-silent"))), c["ab"])
        ck.eq(f"t={t} m-false", int(num(tx("m-false"))), c["ba"])
        ck.eq(f"t={t} m-benrej", int(num(tx("m-benrej"))), c["bu"])
        rows = page.eval_on_selector_all(
            "#fam tbody tr", "trs => trs.map(tr => [...tr.cells].map(td => td.textContent))"
        )
        got = {cells[0]: [int(x) for x in cells[1:]] for cells in rows}
        want = {classes[k]: v for k, v in r["fam"].items()}
        ck.eq(f"t={t} per-family table", got, want)
        if t == 0.0:
            for name, block in per_family["confusion_full_coverage"].items():
                called = block["called"]
                other = sum(v for k, v in called.items() if k not in (name, "benign"))
                ck.eq(f"t=0 {name} vs per_family.json",
                      got[name][:4], [block["n_flows"], called.get(name, 0), other,
                                      called.get("benign", 0) if name != "benign" else 0])


def check_zero_day(page, ck: Checker, z) -> None:
    rare = z["summary"]["rare_families"]
    thresholds = z["config"]["reject_thresholds"]

    def at(rounds, fam, t):
        r = next(r for r in rounds if r["held_out_family"] == fam)
        return next(b for b in r["arms"]["hierarchy"]["by_threshold"] if b["threshold"] == t)

    def mean(fams, t, get, rounds=None):
        return sum(get(at(rounds or z["rounds"], f, t)) for f in fams) / len(fams)

    options = page.eval_on_selector_all("#fam-pick option", "os => os.map(o => o.value)")
    ck.eq("family options", sorted(options),
          sorted(["__mean", *rare, *[r["held_out_family"] for r in z["rounds"]
                                     if r["held_out_family"] not in rare]]))
    for opt in options:
        page.select_option("#fam-pick", opt)
        fams = rare if opt == "__mean" else [opt]
        for i, t in enumerate(thresholds):
            page.locator("#zt button").nth(i).click()
            tx = lambda i_: page.text_content(f"#{i_}") or ""  # noqa: E731
            where = f"held-out {opt} at {t}"
            for cell, key in [("zn-unknown", "rejected_unknown"), ("zn-benign", "called_benign"),
                              ("zn-wrong", "called_wrong_attack")]:
                ck.eq(f"{where} {cell}", tx(cell),
                      zpct(mean(fams, t, lambda b, k=key: b["unseen_family"][k])))
            cov = mean(fams, t, lambda b: b["seen_families"]["coverage"])
            rel = mean(fams, t, lambda b: b["seen_families"]["reliability"])
            ck.eq(f"{where} known traffic", tx("z-seen"), f"{zpct(cov)} at {zpct(rel)}")
            lift = mean(fams, t, lambda b: b["novelty_lift"])
            ctrl = mean(fams, t, lambda b: b["novelty_lift"], z["shuffled_label_control"])
            ck.near(f"{where} novelty lift", num(tx("z-lift")), lift, 0.0005)
            ck.near(f"{where} shuffled control", num(tx("z-ctrl")), ctrl, 0.0005)
            n = sum(at(z["rounds"], f, t)["unseen_family"]["n"] for f in fams)
            m = re.search(r"([\d,]+) unseen flows", tx("z-note"))
            ck.eq(f"{where} flow count", int(m.group(1).replace(",", "")) if m else None, n)


def check_headline(page, ck: Checker, demo, metrics, z) -> None:
    s = z["summary"]
    f0 = s["forced_to_answer_threshold_0"]["per_arm"]["hierarchy"]
    h = s["at_threshold_0.99"]["per_arm"]["hierarchy"]
    got = nums(page.text_content("#h-silent") or "")
    ck.near("headline forced called-benign", got[0], 100 * f0["mean_called_benign"], 0.05)
    ck.near("headline 0.99 called-benign", got[1], 100 * h["mean_called_benign"], 0.05)
    ck.near("headline rejected unknown", num(page.text_content("#h-unknown") or ""),
            100 * h["mean_rejected_unknown"], 0.05)
    row = next(r for r in metrics["coverage_reliability_curve"] if r["threshold"] == 0.99)
    got = nums(page.text_content("#h-cov") or "")
    ck.near("headline coverage", got[0], 100 * row["coverage"], 0.05)
    ck.near("headline reliability", got[1], 100 * row["reliability"], 0.05)
    r = recount(demo, 0.99)["c"]
    n_benign = r["ba"] + r["bb"] + r["bu"]
    got = nums(page.text_content("#h-cov-ci") or "")
    # "...83.2% right. At 0.99, 1,824 of 3,088 benign flows (59.1%) go to an analyst;
    # the 2,315-flow queue is 78.8% benign."
    ck.near("headline full-coverage reliability", got[0],
            100 * metrics["coverage_reliability_curve"][0]["reliability"], 0.05)
    ck.eq("headline benign sent to analyst", (got[2], got[3]), (r["bu"], n_benign))
    ck.eq("headline analyst queue size", got[5], r["unk"])
    ck.near("headline queue benign share", got[6], 100 * r["bu"] / r["unk"], 0.05)


def serve(root: Path) -> tuple[ThreadingHTTPServer, str]:
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def run(root: Path, screenshots: Path | None) -> int:
    from playwright.sync_api import sync_playwright

    demo, z, metrics = (json.loads((ART / n).read_text()) for n in DATA_FILES)
    per_family = json.loads((ART / "per_family.json").read_text())
    ck = Checker()
    httpd, url = serve(root)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for width, height in [(390, 844), (1280, 900)]:
                page = browser.new_page(viewport={"width": width, "height": height})
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.on("console", lambda m, errors=errors: m.type == "error"
                        and errors.append(m.text))
                page.goto(url)
                page.wait_for_function("document.getElementById('n-right').textContent !== ''")
                page.wait_for_function("document.getElementById('zn-unknown').textContent !== ''")
                ck.eq(f"{width}px load error banner", page.locator(".err").count(), 0)
                if width == 390:
                    check_headline(page, ck, demo, metrics, z)
                    check_knob(page, ck, demo, metrics, per_family)
                    check_zero_day(page, ck, z)
                    set_knob(page, 0.99)
                over = page.evaluate(
                    "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                ck.eq(f"{width}px horizontal overflow (px)", over, 0)
                if screenshots:
                    screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshots / f"flowsentry_{width}.png"),
                                    full_page=True)
                ck.eq(f"{width}px JavaScript errors", errors, [])
                page.close()
            browser.close()
    finally:
        httpd.shutdown()
    for f in ck.failures:
        print(f"FAIL: {f}")
    if ck.failures:
        print(f"{len(ck.failures)} of {ck.checks} page checks failed")
        return 1
    print(f"PASS: {ck.checks} checks of what the page displays, against the committed artifacts")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headless check of the live demo page.")
    ap.add_argument("--build", type=Path, help="only assemble the site into this directory")
    ap.add_argument("--screenshots", type=Path, help="save full-page screenshots here")
    args = ap.parse_args(argv)
    if args.build:
        build_site(args.build)
        print(f"[site ] assembled {args.build}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        return run(build_site(Path(tmp)), args.screenshots)


if __name__ == "__main__":
    sys.exit(main())
