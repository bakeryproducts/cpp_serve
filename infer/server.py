import argparse

from flask import Flask, jsonify, request

from infer import get_infer, get_model, load_to_tensor, _log

app = Flask(__name__)


@app.route("/infer/<string:device>/<string:mode>", methods=['POST'])
def post(device, mode):
    """ Recieve post request with image, do inference on that image

    Args:
        device (string): gpu / cpu / trt different devices to do inference on
        mode (string): single / bench_<int> single image run or
          benchmarking multiple forward passes

    Returns:
        string: string with json of prediction results from NN
    """    
    res = dict()
    try:
        model = get_model(device)
        infer = get_infer(mode)

        img = load_to_tensor(request.files['fileupload'])
        res = infer(img, model)
    except Exception as e:
        res[f'POST_HANDLE_ERROR_{__file__}'] = str(e)

    s = jsonify(res)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', const=True, default=False, nargs='?', help='debug')
    args = parser.parse_args()
    host, port = '0.0.0.0', 5000
    log = _log
    app.run(host=host, port=port, debug=args.d)
