"""Editor-side: run Python sent over TCP 127.0.0.1:18801 on the Slate tick (game thread); reply with its stdout.
scripts/start_unreal.sh starts the editor with it (-ExecCmds "py <this file>"); the client is
locomani.sim.unreal.editor (uecmd). It also sets the render default the host project's ini cannot: the editor
viewport off (ShowFlag.* are ECVF_Cheat cvars, refused from project inis). ShowFlag.* reach only editor and game
viewports, not URLab's scene captures; with the viewport off the idle GPU drops from 28 to 11 %, and the captures'
indirect light is ~2/255 darker, the same convergence (TODO 12.7.5-12.7.6)."""

import contextlib
import io
import socket
import traceback

import unreal

_SRV = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_SRV.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_SRV.bind(("127.0.0.1", 18801))
_SRV.listen(4)
_SRV.setblocking(False)
_END = b"\n__END__\n"


def _tick(_dt):
    try:
        conn, _ = _SRV.accept()
    except (BlockingIOError, OSError):
        return
    conn.setblocking(True)
    conn.settimeout(30)
    data = b""
    while not data.endswith(_END):
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        data += chunk
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(data[: -len(_END)].decode(), {"unreal": unreal, "__name__": "__uecmd__"})
    except Exception:
        buf.write(traceback.format_exc())
    conn.sendall(buf.getvalue().encode() or b"(no output)\n")
    conn.close()


_HANDLE = unreal.register_slate_post_tick_callback(_tick)
# The editor viewport off (see the docstring); editor.require_render_defaults refuses to render without it.
unreal.SystemLibrary.execute_console_command(None, "ShowFlag.Rendering 0")
unreal.log("[uecmd] listening on 127.0.0.1:18801")
