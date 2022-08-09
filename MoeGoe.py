from urllib.parse import unquote
from torch import no_grad, LongTensor
import logging
# -*- coding: utf8 -*-
logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence

from scipy.io.wavfile import write

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps_ms.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def print_speakers(speakers):
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        print(str(id) + '\t' + name)

model = '243_epochs.pth'
config = 'config.json'
hps_ms = utils.get_hparams_from_file(config)
net_g_ms = SynthesizerTrn(
    len(hps_ms.symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g_ms.eval()
_ = utils.load_checkpoint(model, net_g_ms, None)
net_g_ms = net_g_ms.cuda()

import io

def speak(speaker: int, text: str) -> bytes:
    stn_tst = get_text(text, hps_ms)

    out_path = io.BytesIO()

    with no_grad():
        x_tst = stn_tst.unsqueeze(0).cuda()
        x_tst_lengths = LongTensor([stn_tst.size(0)]).cuda()
        sid = LongTensor([speaker]).cuda()
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(out_path, hps_ms.data.sampling_rate, audio)
    out_path.seek(0)
    return out_path.read()

from http.server import HTTPServer, BaseHTTPRequestHandler

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split('/')
        try:
            if p[0]: return
            speaker = int(p[1])
            text = unquote(p[2])
        except:
            return
        print(speaker)
        print(text)
        self.send_response(200)
        self.send_header('Content-type', 'audio/x-wave')
        self.end_headers()
        self.wfile.write(speak(speaker, text))
    
server = HTTPServer(('0.0.0.0', 35465), MyHandler)
server.serve_forever()