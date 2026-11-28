"""Private loopback speech-api-v1 server for managed Qwen models."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import sys
import wave


class QwenProvider:
    def __init__(self, spec, models, role):
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        torch.set_num_threads(min(8,torch.get_num_threads()))
        folder = models/spec["model"]
        size = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
        self.device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>size*1.15+512*1024**2 else "cpu"
        options=dict(device_map=self.device,dtype=torch.float32 if self.device=="cpu" else torch.bfloat16,local_files_only=True)
        if role=="asr":
            self.model=Qwen3ASRModel.from_pretrained(str(folder),**options,max_inference_batch_size=1,max_new_tokens=2048)
        else:
            self.model=Qwen3ForcedAligner.from_pretrained(str(folder),**options)
            self.languages=set(self.model.get_supported_languages() or [])

    def recognize(self,clip,language,native):
        result=self.model.transcribe(audio=clip,language=language)[0]
        return dict(text=result.text,language=result.language)

    def align(self,clip,language,text):
        if self.languages and language.lower() not in self.languages:
            return {"alignment":"unsupported_language","timestamps":[]}
        tokens=self.model.align(audio=clip,text=text,language=language)[0]
        return {"timestamps":[dict(text=t.text,start=t.start_time,end=t.end_time) for t in tokens]}


def decode_audio(body):
    import numpy as np
    if "audio_f32le_base64" in body:
        if body.get("sample_rate")!=16000:
            raise ValueError("Expected 16 kHz audio")
        samples=np.frombuffer(base64.b64decode(body["audio_f32le_base64"],validate=True),dtype="<f4")
        if not 0 < samples.size <= 16000*121 or not np.isfinite(samples).all():
            raise ValueError("Invalid float audio")
        return samples.copy(),16000
    data = base64.b64decode(body["audio_wav_base64"], validate=True)
    with wave.open(io.BytesIO(data), "rb") as audio:
        count = audio.getnframes()
        if audio.getparams()[:3] != (1, 2, 16000) or not 0 < count <= 16000 * 121:
            raise ValueError("Invalid mono PCM16 audio window")
        samples = np.frombuffer(audio.readframes(count), dtype="<i2")
        if samples.size != count:
            raise ValueError("Incomplete audio")
    return samples.astype(np.float32) / 32767.0, 16000


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.connection.settimeout(15)
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.send_error(403)
            return
        role = {"/transcribe":"asr", "/align":"aligner"}.get(self.path)
        if role not in self.server.roles:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 12*1024*1024:
                raise ValueError("Unexpected request size")
            body = json.loads(self.rfile.read(size))
            spec = self.server.roles[role]
            language = body.get("language")
            if (body["schema"] != "speech-api-v1" or body["role"] != self.path[1:]
                    or body["model"] != spec["model"] or body["revision"] != spec["manifest_sha256"]):
                raise ValueError("Unexpected role or model")
            if language is not None and not isinstance(language,str):
                raise ValueError("Invalid language")
            if role == "aligner" and (not language or not isinstance(body.get("text"),str)):
                raise ValueError("Alignment requires language and text")
            clip = decode_audio(body)
        except (KeyError,ValueError,TypeError,EOFError,wave.Error):
            self.send_error(400)
            return
        try:
            if role not in self.server.models:
                self.server.models[role] = QwenProvider(spec,self.server.folder,role)
            model = self.server.models[role]
            result = model.recognize(clip,language,False) if role=="asr" else model.align(clip,language,body["text"])
            result["device"] = model.device
            output = json.dumps(result,ensure_ascii=False).encode()
        except Exception as exc:
            print(f"Local speech inference failed ({role}): {type(exc).__name__}",file=sys.stderr,flush=True)
            self.send_error(500)
            return
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(output)))
        self.end_headers()
        self.wfile.write(output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ready",required=True,type=Path)
    args=parser.parse_args()
    config=json.loads(sys.stdin.buffer.readline(65536))
    with HTTPServer(("127.0.0.1",0),Handler) as server:
        server.token=config["token"]
        server.roles=config["roles"]
        server.folder=Path(config["folder"])
        server.models={}
        port_file=args.ready.with_suffix(".tmp")
        port_file.write_text(str(server.server_port),encoding="ascii")
        port_file.replace(args.ready)
        server.timeout=.25
        while args.ready.exists():
            server.handle_request()


if __name__=="__main__":
    main()
