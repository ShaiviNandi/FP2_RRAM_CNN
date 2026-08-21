#!/usr/bin/env python3
"""
Local backend for the ReRAM crossbar visualizer. Imports column_mac_test.py
and crossbar_array_test.py DIRECTLY -- every number this server returns
comes from the same validated code used throughout the SPICE debugging
session, not a reimplementation.

Two execution engines, selectable per-request:
  - "golden": fast, pure-Python nodal-analysis model. No ngspice needed.
  - "ngspice": generates the real dynamic program-then-read netlist and
    runs it through actual ngspice. Slower (real transient simulation),
    but this is actual device physics, not an approximation.

Run:
    pip install -r requirements.txt
    python3 server.py --osdi /path/to/rram_v_1_0_0.osdi --ngspice-bin /usr/local/bin/ngspice
    # then open http://localhost:5057 in a browser
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import column_mac_test as col
import crossbar_array_test as cb

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

CONFIG = {
    "osdi_path": os.environ.get("RRAM_OSDI_PATH", "rram_v_1_0_0.osdi"),
    "ngspice_bin": os.environ.get("NGSPICE_BIN", "ngspice"),
    "ngspice_timeout_s": 180,
}


def _ngspice_available():
    path = shutil.which(CONFIG["ngspice_bin"]) or (CONFIG["ngspice_bin"] if os.path.exists(CONFIG["ngspice_bin"]) else None)
    return path is not None, path


def _osdi_available():
    return os.path.exists(CONFIG["osdi_path"])


@app.route("/")
def index():
    """index.html lives next to this file, not in static/. The original
    send_from_directory(app.static_folder, ...) 404s unless a static/ dir
    exists, which it does not in this repo -- serve from whichever is
    present so the GUI works out of the box."""
    for d in (os.path.dirname(os.path.abspath(__file__)), app.static_folder):
        if d and os.path.exists(os.path.join(d, "index.html")):
            return send_from_directory(d, "index.html")
    return ("index.html not found next to server.py", 404)


# ---------------------------------------------------------------------------
# Vendored frontend dependencies
# index.html used to <script src> these straight from unpkg. When that is
# blocked or offline the browser renders a blank white page and says nothing,
# because a failed script tag is silent. Serving them through here means the
# first successful load caches them to ./vendor/ and every later run works
# offline; uncached assets redirect to the CDN so nothing breaks
# for someone who does have internet.
# ---------------------------------------------------------------------------
VENDOR_FILES = {
    "react.js": "https://unpkg.com/react@18/umd/react.production.min.js",
    "react-dom.js": "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    "babel.js": "https://unpkg.com/@babel/standalone/babel.min.js",
    "tailwind.js": "https://cdn.tailwindcss.com",
}


def _vendor_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    os.makedirs(d, exist_ok=True)
    return d


def fetch_vendor(verbose=True):
    """Download every frontend dependency into ./vendor/. Returns (ok, failed)."""
    import urllib.request
    ok, failed = [], []
    for name, url in VENDOR_FILES.items():
        dest = os.path.join(_vendor_dir(), name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) < 1000:
                raise ValueError(f"suspiciously small response ({len(data)} bytes)")
            with open(dest, "wb") as f:
                f.write(data)
            ok.append(name)
            if verbose:
                print(f"  vendored {name:<16} {len(data)/1024:8.1f} KB")
        except Exception as e:
            failed.append((name, str(e)))
            if verbose:
                print(f"  FAILED   {name:<16} {e}")
    return ok, failed


@app.route("/vendor/<path:name>")
def vendor(name):
    if name not in VENDOR_FILES:
        return ("unknown vendor file", 404)
    d = _vendor_dir()
    path = os.path.join(d, name)
    if not os.path.exists(path):
        try:
            import urllib.request
            req = urllib.request.Request(VENDOR_FILES[name],
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            # No cache and no internet from the server -- let the browser try
            # the CDN itself; it may have connectivity the server lacks.
            from flask import redirect
            return redirect(VENDOR_FILES[name], code=302)
    return send_from_directory(d, name, mimetype="application/javascript")


@app.route("/api/status")
def status():
    ng_ok, ng_path = _ngspice_available()
    return jsonify({
        "ngspice_found": ng_ok,
        "ngspice_path": ng_path or CONFIG["ngspice_bin"],
        "osdi_found": _osdi_available(),
        "osdi_path": CONFIG["osdi_path"],
        "r_lrs": col.R_FOR_MAGNITUDE[1.0],
        "r_mid": col.R_FOR_MAGNITUDE[0.5],
        "r_hrs": col.R_FOR_MAGNITUDE[0.0],
        "width_mid_ns": col.WIDTH_FOR_MAGNITUDE[0.5],
        "width_hrs_ns": col.WIDTH_FOR_MAGNITUDE[0.0],
    })


def _run_ngspice(netlist_text, workdir, results_filename):
    netlist_path = os.path.join(workdir, "run.sp")
    with open(netlist_path, "w") as f:
        f.write(netlist_text)
    ng_ok, ng_path = _ngspice_available()
    if not ng_ok:
        return None, f"ngspice not found (looked for '{CONFIG['ngspice_bin']}')"
    t0 = time.time()
    try:
        result = subprocess.run(
            [CONFIG["ngspice_bin"], "-b", netlist_path],
            cwd=workdir, capture_output=True, text=True, timeout=CONFIG["ngspice_timeout_s"],
        )
    except subprocess.TimeoutExpired:
        return None, f"ngspice timed out after {CONFIG['ngspice_timeout_s']}s -- try a smaller M/K or increase --ngspice-timeout"
    elapsed = time.time() - t0
    if result.returncode != 0:
        return None, f"ngspice exited with error (ran {elapsed:.1f}s): {result.stderr[-800:]}"
    results_path = os.path.join(workdir, results_filename)
    if not os.path.exists(results_path):
        return None, f"ngspice ran ({elapsed:.1f}s) but produced no results file. stderr: {result.stderr[-500:]}"
    return results_path, None


def _digital_exact(W, activations, topology):
    M, K = len(activations), len(W[0])
    if topology == "2t2r":
        return [sum(activations[i] * cb.quantize_to_fp2(W[i][k]) for i in range(M)) for k in range(K)]
    else:
        return [sum(activations[i] * max(cb.quantize_to_fp2(W[i][k]), 0.0) for i in range(M)) for k in range(K)]


@app.route("/api/matmul", methods=["POST"])
def api_matmul():
    body = request.get_json(force=True)
    W = body["weights"]
    activations = body["activations"]
    topology = body.get("topology", "2t2r")
    engine = body.get("engine", "golden")
    r_sense = float(body.get("r_sense", 20.0))
    vread = float(body.get("vread", 0.1))
    K = len(W[0])
    digital = _digital_exact(W, activations, topology)

    if engine == "golden":
        if topology == "2t2r":
            golden = cb.golden_array_matmul(W, activations, r_sense, vread)
        else:
            golden = cb.golden_array_matmul_1t1r(W, activations, r_sense, vread)
        g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
        analog = [g / (vread * g_lrs) for g in golden]
        return jsonify({"engine": "golden", "digital": digital, "analog": analog,
                         "cell_count": (2 if topology == "2t2r" else 1) * len(activations) * K})

    # engine == "ngspice": real dynamic simulation
    osdi_path = os.path.abspath(CONFIG["osdi_path"])
    workdir = tempfile.mkdtemp(prefix="crossbar_ngspice_")
    try:
        slot_duration = float(body.get("slot_duration", 240.0))
        switch_settle_ns = float(body.get("switch_settle_ns", 0.04))
        results_filename = "array_dynamic_results.txt" if topology == "2t2r" else "array_1t1r_dynamic_results.txt"
        if topology == "2t2r":
            netlist = cb.build_array_program_and_read_netlist(
                osdi_path, W, activations, r_sense, vread, slot_duration, switch_settle_ns, results_filename)
        else:
            netlist = cb.build_array_program_and_read_netlist_1t1r(
                osdi_path, W, activations, r_sense, vread, slot_duration, switch_settle_ns, results_filename)

        results_path, err = _run_ngspice(netlist, workdir, results_filename)
        if err:
            return jsonify({"engine": "ngspice", "error": err, "netlist_preview": netlist[:1500]}), 500

        # last row of the results file = settled final read values
        with open(results_path) as f:
            lines = [l for l in f if l.strip()]
        last = lines[-1].split()
        vals = [float(last[j]) for j in range(1, len(last), 2)]
        if topology == "2t2r":
            analog = [(vals[2 * k] - vals[2 * k + 1]) / r_sense for k in range(K)]
        else:
            analog = [vals[k] / r_sense for k in range(K)]
        g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
        analog_recovered = [a / (vread * g_lrs) for a in analog]
        return jsonify({"engine": "ngspice", "digital": digital, "analog": analog_recovered,
                         "cell_count": (2 if topology == "2t2r" else 1) * len(activations) * K,
                         "netlist": netlist})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.route("/api/conv", methods=["POST"])
def api_conv():
    body = request.get_json(force=True)
    kernels = body["kernels"]
    input_fmap = body["input_fmap"]
    topology = body.get("topology", "2t2r")
    engine = body.get("engine", "golden")
    r_sense = float(body.get("r_sense", 20.0))
    vread = float(body.get("vread", 0.1))
    stride = int(body.get("stride", 1))

    W = cb.kernels_to_weight_matrix(kernels)
    kh, kw = len(kernels[0][0]), len(kernels[0][0][0])
    patches, out_h, out_w = cb.im2col_patches(input_fmap, kh, kw, stride)
    K = len(kernels)

    if engine == "golden":
        outputs = []
        for p in patches:
            digital = _digital_exact(W, p, topology)
            if topology == "2t2r":
                golden = cb.golden_array_matmul(W, p, r_sense, vread)
            else:
                golden = cb.golden_array_matmul_1t1r(W, p, r_sense, vread)
            g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
            analog = [g / (vread * g_lrs) for g in golden]
            outputs.append({"digital": digital, "analog": analog})
        return jsonify({"engine": "golden", "out_h": out_h, "out_w": out_w, "outputs": outputs,
                         "cell_count": (2 if topology == "2t2r" else 1) * len(W) * K})

    osdi_path = os.path.abspath(CONFIG["osdi_path"])
    workdir = tempfile.mkdtemp(prefix="crossbar_ngspice_conv_")
    try:
        slot_duration = float(body.get("slot_duration", 240.0))
        switch_settle_ns = float(body.get("switch_settle_ns", 0.04))
        read_window_ns = float(body.get("read_window_ns", 2.0))
        results_filename = "conv_dynamic_results.txt" if topology == "2t2r" else "conv_1t1r_dynamic_results.txt"
        if topology == "2t2r":
            netlist, rwt, out_h2, out_w2, K2 = cb.build_conv_dynamic_netlist(
                osdi_path, kernels, input_fmap, stride, r_sense, vread, slot_duration, switch_settle_ns,
                read_window_ns, results_filename)
        else:
            netlist, rwt, out_h2, out_w2, K2 = cb.build_conv_dynamic_netlist_1t1r(
                osdi_path, kernels, input_fmap, stride, r_sense, vread, slot_duration, switch_settle_ns,
                read_window_ns, results_filename)

        results_path, err = _run_ngspice(netlist, workdir, results_filename)
        if err:
            return jsonify({"engine": "ngspice", "error": err, "netlist_preview": netlist[:1500]}), 500

        differential = topology == "2t2r"
        parsed = cb.parse_conv_dynamic_results_generic(results_path, rwt, K2, r_sense, differential)
        g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
        outputs = []
        for p_idx, p in enumerate(patches):
            digital = _digital_exact(W, p, topology)
            vals = parsed[p_idx]
            analog = [v / (vread * g_lrs) for v in vals] if vals else [None] * K
            outputs.append({"digital": digital, "analog": analog})
        return jsonify({"engine": "ngspice", "out_h": out_h, "out_w": out_w, "outputs": outputs,
                         "cell_count": (2 if topology == "2t2r" else 1) * len(W) * K, "netlist": netlist})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# =============================================================================
# ACCELERATOR DASHBOARD API
# Everything below serves the "Accelerator" tab of index.html: it exposes the
# CSV/JSON artifacts produced by benchmark_resnet18.py, qat_finetune_fp2.py,
# ngspice_full_sweep.py and hw_model.py, launches those scripts as background
# jobs with streamable logs, and runs the interactive crossbar sweep.
# =============================================================================
import csv as _csv
import math as _math
import re as _re
import threading
import uuid
from collections import deque

WORKDIR = os.path.dirname(os.path.abspath(__file__))
JOBS = {}
JOBS_LOCK = threading.Lock()

# Only these scripts can be launched, with only these flags. The dashboard
# builds command lines from user input, so an allowlist is the difference
# between a local tool and a remote-code-execution endpoint. Flag VALUES are
# still passed through, so they are type-checked below before use.
ALLOWED_SCRIPTS = {
    "benchmark_resnet18.py", "qat_finetune_fp2.py",
    "ngspice_full_sweep.py", "hw_model.py",
}
_SAFE_ARG = _re.compile(r"^[A-Za-z0-9_./=+-]*$")


def _csv_rows(path):
    with open(path, newline="") as f:
        return [dict(r) for r in _csv.DictReader(f)]


def _numify(rows):
    """CSV values arrive as strings; coerce anything numeric so the frontend
    can plot without re-parsing, leaving genuine strings (layer names) alone.
    NaN is mapped to None because JSON has no NaN and json.dumps emitting a
    bare NaN token breaks strict parsers including the browser's."""
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                fv = float(v)
                d[k] = None if _math.isnan(fv) or _math.isinf(fv) else fv
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


@app.route("/api/artifacts")
def api_artifacts():
    """Everything on disk the dashboard knows how to render."""
    out = {"csv": [], "json": [], "checkpoints": []}
    for fn in sorted(os.listdir(WORKDIR)):
        p = os.path.join(WORKDIR, fn)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        info = {"name": fn, "size": st.st_size, "mtime": st.st_mtime}
        if fn.endswith(".csv"):
            try:
                with open(p, newline="") as f:
                    rd = _csv.reader(f)
                    header = next(rd, [])
                    info["columns"] = header
                    info["rows"] = sum(1 for _ in rd)
                out["csv"].append(info)
            except Exception:
                pass
        elif fn.endswith(".json"):
            out["json"].append(info)
        elif fn.endswith(".pth"):
            out["checkpoints"].append(info)
    return jsonify(out)


@app.route("/api/csv/<path:name>")
def api_csv(name):
    if "/" in name or "\\" in name or not name.endswith(".csv"):
        return jsonify({"error": "bad name"}), 400
    p = os.path.join(WORKDIR, name)
    if not os.path.exists(p):
        return jsonify({"error": f"{name} not found"}), 404
    return jsonify({"name": name, "rows": _numify(_csv_rows(p))})


@app.route("/api/json/<path:name>")
def api_json(name):
    if "/" in name or "\\" in name or not name.endswith(".json"):
        return jsonify({"error": "bad name"}), 400
    p = os.path.join(WORKDIR, name)
    if not os.path.exists(p):
        return jsonify({"error": f"{name} not found"}), 404
    with open(p) as f:
        return jsonify({"name": name, "data": json.load(f)})


# ---------------------------------------------------------------------------
# Interactive crossbar sweep
# ---------------------------------------------------------------------------
def _sweep_point(M, K, r_sense, vread, utilization, seed):
    """SNR of the analog crossbar vs the digital-exact FP2 dot product for one
    (M, K, r_sense) operating point.

    NOTE ON WHAT THIS IS: the weight matrix is SYNTHETIC -- random FP2 levels
    at the requested utilization -- and the activations are uniform random.
    That makes the sweep instant and dependency-free, and the SHAPE of the
    r_sense / M curves is governed by the resistive-divider physics rather
    than by which particular weights are loaded, so the trends transfer. The
    absolute dB values do NOT transfer to a real layer: real activations are
    sparse and heavy-tailed post-ReLU. For real per-layer numbers, launch
    benchmark_resnet18.py from the Jobs panel."""
    import random as _random
    rng = _random.Random(seed)
    sparsity = max(0.0, 1.0 - utilization)
    W = cb.random_fp2_matrix(M, K, seed=seed, sparsity=sparsity)
    acts = [rng.uniform(-1, 1) for _ in range(M)]

    digital = [sum(acts[i] * cb.quantize_to_fp2(W[i][k]) for i in range(M)) for k in range(K)]
    golden = cb.golden_array_matmul(W, acts, r_sense, vread)
    g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
    analog = [g / (vread * g_lrs) for g in golden]

    err = [a - d for a, d in zip(analog, digital)]
    sig = sum(d * d for d in digital) / K
    noi = sum(e * e for e in err) / K
    snr = 10.0 * _math.log10(sig / noi) if noi > 1e-30 and sig > 1e-30 else float("inf")
    den = sum(abs(d) for d in digital)
    rel = 100.0 * sum(abs(e) for e in err) / den if den > 1e-15 else float("nan")
    nz = sum(1 for r in W for w in r if cb.quantize_to_fp2(w) != 0)
    return {
        "M": M, "K": K, "r_sense": r_sense, "vread": vread,
        "snr_db": None if _math.isinf(snr) else snr,
        "rel_err_pct": None if _math.isnan(rel) else rel,
        "cell_util_pct": 100.0 * nz / (M * K),
    }


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    b = request.get_json(force=True)
    var = b.get("variable", "r_sense")
    M = int(b.get("M", 32)); K = int(b.get("K", 16))
    r_sense = float(b.get("r_sense", 20.0)); vread = float(b.get("vread", 0.1))
    util = float(b.get("utilization", 0.42)); seed = int(b.get("seed", 0))
    n_seeds = max(1, min(int(b.get("n_seeds", 3)), 16))

    if var == "r_sense":
        values = b.get("values") or [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    elif var == "M":
        values = b.get("values") or [8, 16, 32, 64, 128, 256]
    elif var == "utilization":
        values = b.get("values") or [0.1, 0.2, 0.3, 0.42, 0.55, 0.7, 0.85, 1.0]
    else:
        return jsonify({"error": f"unknown variable {var!r}"}), 400

    pts = []
    for v in values:
        acc = []
        for s in range(n_seeds):
            kw = dict(M=M, K=K, r_sense=r_sense, vread=vread, utilization=util, seed=seed + s)
            kw[var if var != "utilization" else "utilization"] = v
            if var == "M":
                kw["M"] = int(v)
            acc.append(_sweep_point(**kw))
        snrs = [p["snr_db"] for p in acc if p["snr_db"] is not None]
        pts.append({
            "value": v,
            "snr_db": sum(snrs) / len(snrs) if snrs else None,
            "snr_min": min(snrs) if snrs else None,
            "snr_max": max(snrs) if snrs else None,
            "rel_err_pct": sum(p["rel_err_pct"] for p in acc) / len(acc),
            "cell_util_pct": sum(p["cell_util_pct"] for p in acc) / len(acc),
        })
    return jsonify({"variable": var, "points": pts, "n_seeds": n_seeds,
                    "synthetic": True})


# ---------------------------------------------------------------------------
# Hardware model
# ---------------------------------------------------------------------------
@app.route("/api/hwmodel", methods=["POST"])
def api_hwmodel():
    try:
        import hw_model as hm
    except ImportError as e:
        return jsonify({"error": f"hw_model.py not importable: {e}"}), 500

    b = request.get_json(force=True)
    saved = dict(hm.ASSUMPTIONS)
    try:
        for k, v in (b.get("assumptions") or {}).items():
            if k in hm.ASSUMPTIONS:
                hm.ASSUMPTIONS[k] = float(v)

        layers_csv = b.get("layers_csv")
        if not layers_csv or "/" in layers_csv or "\\" in layers_csv:
            return jsonify({"error": "layers_csv must be a bare filename in the work dir"}), 400
        lp = os.path.join(WORKDIR, layers_csv)
        if not os.path.exists(lp):
            return jsonify({"error": f"{layers_csv} not found"}), 404
        layers = hm.load_layers_csv(lp)

        power = {}
        pc = b.get("power_csv")
        if pc and os.path.exists(os.path.join(WORKDIR, pc)):
            power = hm.load_power_csv(os.path.join(WORKDIR, pc))

        tile_m = int(b.get("tile_m", 32)); tile_k = int(b.get("tile_k", 16))
        phys = int(b.get("physical_tiles", 0) or 0)
        if b.get("sweep_m"):
            ms = b.get("m_values") or [8, 16, 32, 64, 128, 256]
            sweep = []
            for m in ms:
                R = hm.rollup(layers, power, int(m), tile_k, verbose=False,
                              physical_tiles=phys)
                d = hm.delay_model(int(m), tile_k,
                                   sum(L["util"] for L in layers) / len(layers))
                adc = sum(r["adc_energy_frac"] for r in R["per_layer"]) / len(R["per_layer"])
                sweep.append({"tile_m": int(m), "tiles": R["total_tiles"],
                              "t_tile_ns": d["t_tile_ns"], "pj_per_mac": R["pj_per_mac"],
                              "adc_energy_frac": adc, "area_mm2": R["area"]["total_mm2"],
                              "tops_per_w": R["tops_per_w"],
                              "tops_per_mm2": R["tops_per_mm2"]})
            return jsonify({"sweep_m": sweep, "assumptions": hm.ASSUMPTIONS,
                            "power_measured_layers": len(power)})

        R = hm.rollup(layers, power, tile_m, tile_k, verbose=False, physical_tiles=phys)
        d = hm.delay_model(tile_m, tile_k, sum(L["util"] for L in layers) / len(layers))
        return jsonify({"result": R, "delay": d, "assumptions": hm.ASSUMPTIONS,
                        "power_measured_layers": len(power),
                        "total_layers": len(layers)})
    finally:
        hm.ASSUMPTIONS.clear()
        hm.ASSUMPTIONS.update(saved)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
def _job_thread(job_id, cmd):
    job = JOBS[job_id]
    try:
        proc = subprocess.Popen(cmd, cwd=WORKDIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with JOBS_LOCK:
            job["pid"] = proc.pid
        for line in proc.stdout:
            with JOBS_LOCK:
                job["log"].append(line.rstrip("\n"))
                job["lines"] += 1
        proc.wait()
        with JOBS_LOCK:
            job["returncode"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else "failed"
    except Exception as e:
        with JOBS_LOCK:
            job["status"] = "failed"
            job["log"].append(f"[launcher error] {e}")
    finally:
        with JOBS_LOCK:
            job["finished"] = time.time()


@app.route("/api/jobs", methods=["GET", "POST"])
def api_jobs():
    if request.method == "GET":
        with JOBS_LOCK:
            return jsonify({"jobs": [
                {k: v for k, v in j.items() if k != "log"} for j in JOBS.values()
            ]})

    b = request.get_json(force=True)
    script = b.get("script")
    if script not in ALLOWED_SCRIPTS:
        return jsonify({"error": f"script must be one of {sorted(ALLOWED_SCRIPTS)}"}), 400
    args = b.get("args") or []
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        return jsonify({"error": "args must be a list of strings"}), 400
    for a in args:
        if not _SAFE_ARG.match(a):
            return jsonify({"error": f"argument {a!r} contains disallowed characters. "
                                     f"Allowed: letters, digits, _ . / = + -"}), 400

    job_id = uuid.uuid4().hex[:12]
    # sys.executable, not "python3": the server may be running inside a venv
    # whose torch/torchvision are the ones the job needs.
    cmd = [sys.executable, script] + args
    job = {"id": job_id, "script": script, "args": args, "cmd": " ".join(cmd),
           "status": "running", "started": time.time(), "finished": None,
           "returncode": None, "lines": 0, "pid": None,
           "log": deque(maxlen=4000)}
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_job_thread, args=(job_id, cmd), daemon=True).start()
    return jsonify({"id": job_id, "cmd": job["cmd"]})


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "no such job"}), 404
        since = int(request.args.get("since", 0))
        lines = list(job["log"])
        return jsonify({
            "id": job["id"], "script": job["script"], "status": job["status"],
            "returncode": job["returncode"], "started": job["started"],
            "finished": job["finished"], "cmd": job["cmd"],
            "lines": job["lines"],
            "log": lines[since:] if since < len(lines) else [],
            "log_offset": max(since, 0),
        })


@app.route("/api/jobs/<job_id>/stop", methods=["POST"])
def api_job_stop(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or not job.get("pid"):
            return jsonify({"error": "no such running job"}), 404
        pid = job["pid"]
    try:
        os.kill(pid, 15)
        return jsonify({"ok": True})
    except ProcessLookupError:
        return jsonify({"error": "process already gone"}), 404


@app.route("/api/env")
def api_env():
    """What the dashboard can actually offer, given what is installed."""
    ng_ok, ng_path = _ngspice_available()
    def _has(mod):
        try:
            __import__(mod); return True
        except ImportError:
            return False
    return jsonify({
        "workdir": WORKDIR,
        "ngspice": {"found": ng_ok, "path": ng_path},
        "torch": _has("torch"), "torchvision": _has("torchvision"),
        "hw_model": _has("hw_model"),
        "cuda": (__import__("torch").cuda.is_available() if _has("torch") else False),
        "gpu": (__import__("torch").cuda.get_device_name(0)
                if _has("torch") and __import__("torch").cuda.is_available() else None),
        "cpu_count": os.cpu_count(),
        "device_params": {
            "r_lrs": col.R_FOR_MAGNITUDE[1.0],
            "r_mid": col.R_FOR_MAGNITUDE[0.5],
            "r_hrs": col.R_FOR_MAGNITUDE[0.0],
            "on_off_ratio": col.R_FOR_MAGNITUDE[0.0] / col.R_FOR_MAGNITUDE[1.0],
        },
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdi", default=CONFIG["osdi_path"], help="Path to compiled rram_v_1_0_0.osdi")
    ap.add_argument("--ngspice-bin", default=CONFIG["ngspice_bin"], help="Path to ngspice binary")
    ap.add_argument("--ngspice-timeout", type=int, default=CONFIG["ngspice_timeout_s"])
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fetch-vendor", action="store_true",
                    help="Download React/ReactDOM/Babel/Tailwind into ./vendor/ and exit. "
                         "Run this once while online and the GUI works offline forever after.")
    args = ap.parse_args()

    CONFIG["osdi_path"] = args.osdi
    CONFIG["ngspice_bin"] = args.ngspice_bin
    CONFIG["ngspice_timeout_s"] = args.ngspice_timeout

    if args.fetch_vendor:
        print("Downloading frontend dependencies into ./vendor/ ...")
        ok, failed = fetch_vendor()
        print(f"\n{len(ok)} vendored, {len(failed)} failed.")
        raise SystemExit(1 if failed else 0)

    missing_vendor = [n for n in VENDOR_FILES if not os.path.exists(os.path.join(_vendor_dir(), n))]
    if missing_vendor:
        print(f"Frontend deps not cached yet ({', '.join(missing_vendor)}) -- they will be "
              f"fetched on first page load, or run --fetch-vendor now to cache them.")
    else:
        print(f"Frontend deps cached in ./vendor/ ({len(VENDOR_FILES)} files) -- GUI works offline.")

    ng_ok, ng_path = _ngspice_available()
    print(f"ngspice: {'FOUND at ' + ng_path if ng_ok else 'NOT FOUND (golden engine will still work)'}")
    print(f"OSDI model: {'FOUND at ' + os.path.abspath(CONFIG['osdi_path']) if _osdi_available() else 'NOT FOUND (' + CONFIG['osdi_path'] + ') -- ngspice engine will fail until this exists'}")
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
