"""
Build script: pre-renders all Flask pages to static HTML for Netlify deployment.
API endpoints are converted to static JSON files.
"""
import sys
import os
import json
import shutil

# Add the app directory to path
app_dir = os.path.join(os.path.dirname(__file__), 'linear attention models')
sys.path.insert(0, app_dir)

from app import app
from models.s4 import generate_s4_demo_data
from models.mamba1 import generate_selectivity_demo
from models.mamba2 import generate_ssd_demo
from models.delta_net import generate_delta_net_demo

BUILD_DIR = os.path.join(os.path.dirname(__file__), '_site')


def build():
    # Clean
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    # Copy static assets
    static_src = os.path.join(app_dir, 'static')
    static_dst = os.path.join(BUILD_DIR, 'static')
    shutil.copytree(static_src, static_dst)

    # Pre-render HTML pages
    pages = {
        '/': 'index.html',
        '/s4': 's4.html',
        '/mamba1': 'mamba1.html',
        '/mamba2': 'mamba2.html',
        '/delta': 'delta.html',
    }

    with app.test_client() as client:
        for route, filename in pages.items():
            resp = client.get(route)
            filepath = os.path.join(BUILD_DIR, filename)
            # Fix static URLs for Netlify (they should be relative)
            html = resp.data.decode('utf-8')
            # Flask generates /static/... URLs, keep them as-is since we copy static/
            with open(filepath, 'w') as f:
                f.write(html)
            print(f'  Rendered {route} -> {filename} ({len(html)} bytes)')

    # Pre-generate API responses as static JSON
    api_dir = os.path.join(BUILD_DIR, 'api')

    # S4 API - common parameter combos
    for N in [2, 3, 4, 5, 6, 7, 8]:
        for L in [8, 16, 24, 32]:
            d = os.path.join(api_dir, 's4', str(N))
            os.makedirs(d, exist_ok=True)
            data = generate_s4_demo_data(N=N, L=L)
            with open(os.path.join(d, f'{L}.json'), 'w') as f:
                json.dump(data, f)

    # Mamba1 API
    for L in [4, 8, 12, 16]:
        for D in [2, 3, 4, 5, 6]:
            for N in [2, 3, 4, 5]:
                d = os.path.join(api_dir, 'mamba1', str(L), str(D))
                os.makedirs(d, exist_ok=True)
                data = generate_selectivity_demo(B_size=1, L=L, D=D, N=N)
                with open(os.path.join(d, f'{N}.json'), 'w') as f:
                    json.dump(data, f)

    # Mamba2 API
    for T in [4, 6, 8, 10, 12]:
        for N in [2, 3, 4, 5]:
            for P in [1, 2, 3, 4]:
                d = os.path.join(api_dir, 'mamba2', str(T), str(N))
                os.makedirs(d, exist_ok=True)
                data = generate_ssd_demo(T=T, N=N, P=P)
                with open(os.path.join(d, f'{P}.json'), 'w') as f:
                    json.dump(data, f)

    # Delta API
    for T in [4, 6, 8, 10, 12]:
        for dd in [2, 3, 4, 5, 6]:
            for dv in [2, 3, 4, 5]:
                d = os.path.join(api_dir, 'delta', str(T), str(dd))
                os.makedirs(d, exist_ok=True)
                data = generate_delta_net_demo(T=T, d=dd, d_v=dv)
                with open(os.path.join(d, f'{dv}.json'), 'w') as f:
                    json.dump(data, f)

    print(f'\nBuild complete -> {BUILD_DIR}')
    # Count files
    total = sum(len(files) for _, _, files in os.walk(BUILD_DIR))
    print(f'Total files: {total}')


if __name__ == '__main__':
    build()
