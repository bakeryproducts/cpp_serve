import urllib
import argparse
from pathlib import Path
from collections import defaultdict

from flask import Flask, make_response, send_file, jsonify, request, render_template

app = Flask(__name__)


def log(m): print(m)


@app.route('/')
def index():
    s = ''
    s += f'<h2>Hello world</h2>'
    return s



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', const=True, default=False, nargs='?', help='debug')
    args = parser.parse_args()
    host, port = '0.0.0.0', 5000
    app.run(host=host, port=port, debug=args.d)
