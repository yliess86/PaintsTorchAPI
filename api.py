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
    hint[:3] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hint[:3])
    return hint

G  = None
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
    x  = x.replace('data:image/jpeg;base64', '')
    x  = x.replace('data:image/png;base64', '')
    x  = x.replace('data:image/jpg;base64', '')
    x  = x.replace('data:image/tiff;base64', '')
    return x

def colorize(sketch, hint, opacity):
    sketch  = remove_header(sketch)
    hint    = remove_header(hint)

    osketch = Image.open(BytesIO(base64.b64decode(sketch)))
    hint    = Image.open(BytesIO(base64.b64decode(hint)))
    w, h    = osketch.size

    if osketch.mode == 'RGBA':
        bckg    = Image.new('RGB', osketch.size, (255, 255, 255))
        bckg.paste(osketch, mask=osketch.split()[3])
        osketch = bckg

    sketch        = Ts(osketch)
    hint          = Th(hint)

    sketch        = sketch.unsqueeze(0).to(device)
    hint          = hint.unsqueeze(0).to(device)
    colored       = G(sketch, hint, I(sketch)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    colored       = ((colored + 1) * 0.5 * 255.0).astype(np.uint8)

    colored       = Image.fromarray(colored)
    colored       = colored.resize((w, h))

    colored       = np.array(colored.convert('RGBA')).astype(float)
    sketch        = np.array(osketch.convert('RGB')).astype(float)

    bin_a         = (((sketch.mean(2) < 128) * 1.0) * 255.0)
    bin           = np.zeros((colored.shape[0], colored.shape[1], 4))
    bin[:, :, :3] = sketch
    bin[:, :, 3]  = bin_a

    blend         = blend_modes.soft_light(colored, bin, opacity)
    blend         = Image.fromarray(blend.astype(np.uint8))

    buff          = BytesIO()
    blend.save(buff, format='PNG')
    base          = base64.b64encode(buff.getvalue())
    blend         = b'data:image/png;base64,' + base

    return blend

@app.route('/api/v1/colorizer', methods=['POST'])
@cross_origin(origin='*')
def colorizer():
    data = request.json
    if 'sketch' in data and 'hint' in data:
        colored  = colorize(data['sketch'], data['hint'], data['opacity'])
        response = jsonify({'colored': str(colored)[2:-1]})
        return response
    return jsonify({})

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Select the device to run on: either "cpu" and "cuda" are available')
    parser.add_argument('-g', '--generator', type=str, default='./generator.pth', help='Path to generator model')
    parser.add_argument('-i', '--illustration2vec', type=str, default='./i2v.pth', help='Path to Illustration2Vec model')

    args   = parser.parse_args()
    device = args.device

    C = torch.load(args.generator)
    G = Wrapper(Generator(64, in_channels=1, out_channels=3))
    I = Illustration2Vec(path=args.illustration2vec)

    G.load_state_dict(C['generator'])

    G = G.to(device)
    I = I.to(device)

    app.run(debug=True, threaded=True, host='0.0.0.0', port=8888)
