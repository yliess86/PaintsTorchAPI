import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import blend_modes
import base64
import torch
import json
import sys
sys.path.append('/Projects/PaintsTorch/paintstorch')

from flask_cors import CORS, cross_origin
from model import Generator, Illustration2Vec
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

class Wrapper(nn.Module):
    def __init__(self, module):
        super(Wrapper, self).__init__()
        self.module = module

    def forward(self, sketch, hint, features):
        return self.module(sketch, hint, features)

app                         = Flask(__name__)
app.config['CORS_HEADERS']  = 'Content-Type'
app.config['JSON_AS_ASCII'] = False
cors                        = CORS(app)
device                      = 'cpu'

def resize_image_max(x, size):
    m = max(x.width, x.height)
    s = size / m
    w = int(np.floor(x.width * s))
    h = int(np.floor(x.height * s))

    return x.resize((w, h)), w, h

def add_grey(x):
    grey              = np.ones((512 // 4, 512 // 4, 4)) * 128.0
    grey[:, :, 3]    *= 0
    grey              = Image.fromarray(grey.astype(np.uint8))
    mask              = np.array(x)[:, :, 3]

    grey.paste(x, (0, 0), x)

    grey              = np.array(grey)
    grey[:, :, 3]     = mask

    return Image.fromarray(grey)

def normalize_hint(hint):
    hint[:3, ...] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hint[:3, ...])
    return hint

Gs = {}
I  = None
Ts = transforms.Compose([
    lambda x: x.resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    lambda x: x.mean(0).unsqueeze(0)

])
Th = transforms.Compose([
    lambda x: x.resize((512 // 4, 512 // 4)),
    add_grey,
    transforms.ToTensor(),
    normalize_hint
])

def remove_header(x):
    x  = x.replace('data:image/jpeg;base64,', '')
    x  = x.replace('data:image/png;base64,', '')
    x  = x.replace('data:image/jpg;base64,', '')
    x  = x.replace('data:image/tiff;base64,', '')

    x  = x.replace('data:image/jpeg;base64', '')
    x  = x.replace('data:image/png;base64', '')
    x  = x.replace('data:image/jpg;base64', '')
    x  = x.replace('data:image/tiff;base64', '')
    return x

def png2base64(img):
    img  = Image.fromarray(img.astype(np.uint8))
    buff = BytesIO()
    img.save(buff, format='PNG')
    base = b'data:image/png;base64,' + base64.b64encode(buff.getvalue())
    return base

def blend_sketch_colored(sketch, colored, opacity):
    bin_a         = (((sketch.mean(2) < 128) * 1.0) * 255.0)
    bin           = np.zeros((colored.shape[0], colored.shape[1], 4))
    bin[:, :, :3] = sketch
    bin[:, :, 3]  = bin_a
    blend         = blend_modes.soft_light(colored, bin, opacity)
    return blend

def apply_color(s, h, m):
    G       = Gs[m]
    sketch  = Ts(s)
    hint    = Th(h)
    sketch  = sketch.unsqueeze(0).to(device)
    hint    = hint.unsqueeze(0).to(device)
    colored = G(sketch, hint, I(sketch)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    colored = ((colored + 1) * 0.5 * 255.0).astype(np.uint8)
    
    return colored

def colorize(sketch, hint, opacity, model):
    try:
        sketch  = remove_header(sketch)
        hint    = remove_header(hint)

        osketch = Image.open(BytesIO(base64.b64decode(sketch))).convert('RGB')
        hint    = Image.open(BytesIO(base64.b64decode(hint)))
        w, h    = osketch.size

        if osketch.mode == 'RGBA':
            bckg    = Image.new('RGB', osketch.size, (255, 255, 255))
            bckg.paste(osketch, mask=osketch.split()[3])
            osketch = bckg

        colored = apply_color(osketch, hint, model)

        colored = Image.fromarray(colored)
        colored = colored.resize((w, h))
        colored = np.array(colored.convert('RGBA')).astype(float)
        sketch  = np.array(osketch.convert('RGB')).astype(float)

        blend   = blend_sketch_colored(sketch, colored, opacity)
        blend   = png2base64(blend)

        return blend
        
    except Exception as e:
        exception = str(e)
        print('\033[0;31m' + exception + '\033[0m')
        return None

def add_response_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    return response

@app.route('/api/v1/colorizer', methods=['POST'])
@cross_origin(origin='*')
def colorizer():
    try:
        data = request.json

        if 'sketch' in data and 'hint' in data and 'opacity' in data:
            model    = data['model'] if 'model' in data else list(Gs.keys())[0]
            if model not in list(Gs.keys()):
                raise Exception('The model requested does not exists.')

            colored  = colorize(data['sketch'], data['hint'], data['opacity'], model)
            if colored is None:
                raise Exception('There is an issue with the data.')
            
            response = jsonify({ 'success': True, 'colored': str(colored)[2:-1] })
            return response

        else:
            response = jsonify({ 'success': False, 'error': '"sketch" or "hint" or "opacity" may not be included in the json' })
            return response

    except Exception as e:
        exception = str(e)
        print('\033[0;31m' + exception + '\033[0m')
        response  = jsonify({ 'success': False, 'error': exception })
        return response

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Select the device to run on: either "cpu" and "cuda" are available')
    parser.add_argument('-g', '--generators', nargs='+', type=str, help='Path to generator models')
    parser.add_argument('-n', '--names', nargs='+', type=str, help='Name of generator models')
    parser.add_argument('-i', '--illustration2vec', type=str, help='Path to Illustration2Vec model')

    args   = parser.parse_args()
    device = 'cuda:3' if args.device == 'cuda' else args.device

    assert len(args.names) == len(args.generators), f'Should be as many names as generators:\n{args.names}\n{args.generators}'
    for path in args.generators:
        assert os.path.isfile(path), f'File {path} does not exisits'

    for (path, name) in zip(args.generators, args.names):
        C        = torch.load(path)
        Gs[name] = Wrapper(Generator(64, in_channels=1, out_channels=3)) \
                   if args.device == 'cpu' \
                   else nn.DataParallel(Generator(64, in_channels=1, out_channels=3), device_ids=(3, ))
        Gs[name].load_state_dict(C['generator'])
        Gs[name] = Gs[name].to(device)

    I = Illustration2Vec(path=args.illustration2vec) if args.device == 'cpu' else nn.DataParallel(Illustration2Vec(path=args.illustration2vec), device_ids=(3, ))
    I = I.to(device)

    app.run(debug=True, threaded=True, host='0.0.0.0', port=8888)
