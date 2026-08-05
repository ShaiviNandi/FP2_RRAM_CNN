#!/usr/bin/env python3
"""
Local backend for the ReRAM crossbar visualizer. Imports column_mac_test.py
and crossbar_array_test.py DIRECTLY -- every number this server returns
comes from the same validated code used throughout the SPICE debugging
session, not a reimplementation.

Two execution engines, selectable per-request:
  - "golden": fast, pure-Python nodal-analysis model. No ngspice needed.
  - "ngspice": generates the REAL dynamic program-then-read netlist and
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
    return send_from_directory(app.static_folder, "index.html")


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osdi", default=CONFIG["osdi_path"], help="Path to compiled rram_v_1_0_0.osdi")
    ap.add_argument("--ngspice-bin", default=CONFIG["ngspice_bin"], help="Path to ngspice binary")
    ap.add_argument("--ngspice-timeout", type=int, default=CONFIG["ngspice_timeout_s"])
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    CONFIG["osdi_path"] = args.osdi
    CONFIG["ngspice_bin"] = args.ngspice_bin
    CONFIG["ngspice_timeout_s"] = args.ngspice_timeout

    ng_ok, ng_path = _ngspice_available()
    print(f"ngspice: {'FOUND at ' + ng_path if ng_ok else 'NOT FOUND (golden engine will still work)'}")
    print(f"OSDI model: {'FOUND at ' + os.path.abspath(CONFIG['osdi_path']) if _osdi_available() else 'NOT FOUND (' + CONFIG['osdi_path'] + ') -- ngspice engine will fail until this exists'}")
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
