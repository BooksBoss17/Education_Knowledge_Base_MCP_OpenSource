"""Direct HTTPS and Windows physical-interface selection for BiliDown only."""
import ctypes as ct
from http.client import HTTPSConnection
import ipaddress
import os
import socket
import struct
from urllib.request import HTTPSHandler, ProxyHandler, build_opener


class DirectUnavailable(OSError):
    pass


def child_environment():
    clean = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    clean["NO_PROXY"] = "*"
    return clean


class _InterfaceRow(ct.Structure):
    # MIB_IF_ROW2, Windows SDK netioapi.h.
    _fields_ = [
        ("luid", ct.c_uint64), ("index", ct.c_uint32), ("guid", ct.c_byte * 16),
        ("alias", ct.c_wchar * 257), ("description", ct.c_wchar * 257),
        ("address_length", ct.c_uint32), ("addresses", ct.c_byte * 64),
        ("mtu", ct.c_uint32), ("kind", ct.c_uint32), ("tunnel", ct.c_uint32),
        ("media", ct.c_uint32), ("physical_medium", ct.c_uint32),
        ("access", ct.c_uint32), ("direction", ct.c_uint32),
        ("flags", ct.c_ubyte), ("oper", ct.c_uint32), ("admin", ct.c_uint32),
        ("media_state", ct.c_uint32), ("network_guid", ct.c_byte * 16),
        ("connection_type", ct.c_uint32), ("counters", ct.c_uint64 * 20),
    ]


def physical_index(family, address):
    ip = ipaddress.ip_address(address[0])
    if not ip.is_global or ip.is_multicast:
        raise DirectUnavailable("目标地址无法确认为公网直连。")
    sockaddr = struct.pack("=H", family) + b"\0\0"
    if family == socket.AF_INET6:
        sockaddr += bytes(4) + ip.packed + struct.pack("=I", address[3])
    else:
        sockaddr += ip.packed + bytes(8)
    table = ct.windll.iphlpapi
    index = ct.c_uint32()
    error = table.GetBestInterfaceEx(sockaddr, ct.byref(index))
    if error:
        raise OSError(error, "目标地址没有可用路由。")
    row = _InterfaceRow(index=index.value)
    if table.GetIfEntry2(ct.byref(row)) or not row.flags & 1 or row.tunnel or row.oper != 1:
        raise DirectUnavailable("目标路由未使用活动物理网卡，请检查 VPN 分流。")
    return index.value


def pinned_socket(family, kind, proto, address):
    index = physical_index(family, address)
    sock = socket.socket(family, kind, proto)
    try:
        # Windows IP_UNICAST_IF takes network byte order for IPv4, native for IPv6.
        level, order = (socket.IPPROTO_IP, "!I") if family == socket.AF_INET else (socket.IPPROTO_IPV6, "=I")
        sock.setsockopt(level, 31, struct.pack(order, index))
        return sock
    except OSError:
        sock.close()
        raise DirectUnavailable("无法将连接绑定到已核验的物理网卡。") from None


def direct_connect(endpoint, timeout, source_address=None):
    last = None
    blocked = None
    for family, kind, proto, _, address in socket.getaddrinfo(*endpoint, type=socket.SOCK_STREAM):
        sock = None
        try:
            sock = pinned_socket(family, kind, proto, address)
            sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(address)
            return sock
        except OSError as error:
            if sock:
                sock.close()
            last = error
            if isinstance(error, DirectUnavailable):
                blocked = error
    raise blocked or last or DirectUnavailable("没有可用的直连地址。")


def native_source(host):
    """Direct IPv4 source for aria2 (which may open several connections)."""
    choices = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_DGRAM)
    sources = set()
    for family, kind, proto, _, address in choices:
        with pinned_socket(family, kind, proto, address) as sock:
            sock.connect(address)
            sources.add(sock.getsockname()[0])
    if len(sources) != 1:
        raise DirectUnavailable("下载地址没有唯一可绑定的物理出口。")
    return sources.pop()


def check_browser_host(host):
    reachable = False
    for family, _, _, _, address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        try:
            physical_index(family, address)
            reachable = True
        except DirectUnavailable:
            raise
        except OSError:
            continue
    if not reachable:
        raise DirectUnavailable("目标没有可用的物理路由。")


class _DirectHTTPSConnection(HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = direct_connect


class _DirectHTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_DirectHTTPSConnection, request, context=self._context)


def direct_opener(*handlers):
    return build_opener(ProxyHandler({}), _DirectHTTPSHandler(), *handlers)
