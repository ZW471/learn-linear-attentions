"""
Linear Attention Models - Interactive Educational Guide
Flask application serving interactive visualizations of S4, Mamba-1, Mamba-2, and DeltaNet.
"""

from flask import Flask, render_template, jsonify
import json
import numpy as np

from models.s4 import generate_s4_demo_data, generate_s4_execution_steps
from models.mamba1 import generate_selectivity_demo, generate_mamba_block_info, generate_mamba1_execution_steps
from models.mamba2 import generate_ssd_demo, generate_mamba2_execution_steps
from models.delta_net import generate_delta_net_demo, generate_delta_execution_steps

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/s4")
def s4_page():
    demo = generate_s4_demo_data(N=4, L=16)
    exec_steps = generate_s4_execution_steps(N=4, L=8)
    return render_template("s4.html", demo=json.dumps(demo), exec_steps=json.dumps(exec_steps))


@app.route("/mamba1")
def mamba1_page():
    demo = generate_selectivity_demo(B_size=1, L=12, D=4, N=3)
    block_info = generate_mamba_block_info()
    exec_steps = generate_mamba1_execution_steps(L=6, D=3, N=2)
    return render_template("mamba1.html", demo=json.dumps(demo), block_info=block_info,
                           exec_steps=json.dumps(exec_steps))


@app.route("/mamba2")
def mamba2_page():
    demo = generate_ssd_demo(T=8, N=3, P=2)
    exec_steps = generate_mamba2_execution_steps(T=6, N=2, P=2)
    return render_template("mamba2.html", demo=json.dumps(demo), exec_steps=json.dumps(exec_steps))


@app.route("/delta")
def delta_page():
    demo = generate_delta_net_demo(T=8, d=4, d_v=3)
    exec_steps = generate_delta_execution_steps(T=6, d=3, d_v=2)
    return render_template("delta_net.html", demo=json.dumps(demo), exec_steps=json.dumps(exec_steps))


# API endpoints for dynamic parameter changes
@app.route("/api/s4/<int:N>/<int:L>")
def api_s4(N, L):
    N = min(max(N, 2), 16)
    L = min(max(L, 4), 64)
    return jsonify(generate_s4_demo_data(N=N, L=L))


@app.route("/api/mamba1/<int:L>/<int:D>/<int:N>")
def api_mamba1(L, D, N):
    L = min(max(L, 4), 32)
    D = min(max(D, 2), 8)
    N = min(max(N, 2), 8)
    return jsonify(generate_selectivity_demo(B_size=1, L=L, D=D, N=N))


@app.route("/api/mamba2/<int:T>/<int:N>/<int:P>")
def api_mamba2(T, N, P):
    T = min(max(T, 4), 16)
    N = min(max(N, 2), 8)
    P = min(max(P, 1), 4)
    return jsonify(generate_ssd_demo(T=T, N=N, P=P))


@app.route("/api/delta/<int:T>/<int:d>/<int:d_v>")
def api_delta(T, d, d_v):
    T = min(max(T, 4), 16)
    d = min(max(d, 2), 8)
    d_v = min(max(d_v, 2), 8)
    return jsonify(generate_delta_net_demo(T=T, d=d, d_v=d_v))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
