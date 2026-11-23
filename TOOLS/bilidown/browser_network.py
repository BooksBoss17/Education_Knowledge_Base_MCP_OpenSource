"""Short-lived loopback transport for Edge; only Bilibili HTTPS uses physical egress."""
from contextlib import contextmanager
import base64
import hmac
import re
import secrets
import select
import socketserver
import threading

from .direct_network import direct_connect

@contextmanager
def browser_egress():
    key=secrets.token_hex(24)
    expected=base64.b64encode(("search:"+key).encode())

    class Tunnel(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(12)
            try:
                header=self.rfile.readline(4096)
                match=re.fullmatch(rb"CONNECT ([a-z0-9.-]+):443 HTTP/1\.[01]\r\n",header,re.I)
                auth=b"";size=len(header)
                while size<16384:
                    line=self.rfile.readline(4096);size+=len(line)
                    if line in (b"",b"\r\n"):break
                    if line.lower().startswith(b"proxy-authorization: basic "):auth=line.split()[-1]
                if size>=16384:return
                if not hmac.compare_digest(auth,expected):
                    self.wfile.write(b'HTTP/1.1 407 Proxy Authentication Required\r\nProxy-Authenticate: Basic realm="local"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                    return
                host=match[1].decode().lower() if match else ""
                if not any(host==d or host.endswith("."+d) for d in ("bilibili.com","hdslb.com","biliimg.com","biliapi.net","biliapi.com","bilivideo.com")):
                    self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n");return
                with direct_connect((host,443),8) as upstream:
                    self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    peers={upstream:self.request,self.request:upstream}
                    while ready:=select.select(list(peers),[],[],35)[0]:
                        for src in ready:
                            block=src.recv(65536)
                            if not block:return
                            peers[src].sendall(block)
            except (OSError,ValueError):return
    with socketserver.ThreadingTCPServer(("127.0.0.1",0),Tunnel) as server:
        server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:yield {"server":"http://127.0.0.1:"+str(server.server_address[1]),"username":"search","password":key}
        finally:server.shutdown();thread.join()
