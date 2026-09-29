"""The running Unreal editor, from the URLab bridge venv: Python inside it, its level, its render defaults.

The editor (`scripts/start_unreal.sh`) runs `in_editor/cmd_server.py`, which executes Python sent over TCP
127.0.0.1:CMD_PORT on the game thread and replies with its stdout. URLab's own bridge (ZMQ, :5559) moves the
robot and returns the cameras; this module covers what the bridge cannot: loading a level, checking the render
defaults, running the in-editor scripts. One editor per machine: both ports are fixed.

    "$BRIDGE_PY" -m locomani.sim.unreal.editor 'print(unreal.SystemLibrary.get_engine_version())'
    "$BRIDGE_PY" -m locomani.sim.unreal.editor -f src/locomani/sim/unreal/in_editor/no_nanite.py
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

CMD_PORT = 18801
URLAB_HOST = "tcp://127.0.0.1"
IN_EDITOR = Path(__file__).resolve().parent / "in_editor"

_RENDER_DEFAULTS = """
sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = ues.get_game_world() or ues.get_editor_world()
nanite = sorted({c.static_mesh.get_name() for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
                 for c in a.get_components_by_class(unreal.StaticMeshComponent)
                 if c.static_mesh and sub.get_nanite_settings(c.static_mesh).enabled})
cvar = unreal.SystemLibrary.get_console_variable_int_value
print(cvar('r.Shadow.Nanite'), cvar('ShowFlag.Rendering'), len(nanite), ' '.join(nanite[:8]))
"""

_LEVEL = """
ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = ues.get_editor_world()
print(world.get_path_name().split('.')[0] if world else '', bool(ues.get_game_world()))
"""


def uecmd(code: str, timeout: float = 600.0) -> str:
    """Run `code` in the editor's Python (game thread); returns what it printed, a traceback included."""
    try:
        s = socket.create_connection(("127.0.0.1", CMD_PORT), timeout=timeout)
    except OSError as exc:
        raise SystemExit(
            f"no Unreal editor command server on :{CMD_PORT} ({exc}): start the editor, bash scripts/start_unreal.sh"
        ) from exc
    with s:
        s.sendall(code.encode() + b"\n__END__\n")
        out = b""
        while chunk := s.recv(1 << 16):
            out += chunk
    return out.decode()


def run_file(path: Path, **globals_) -> str:
    """Run an in-editor script with module-level names set first (the command server passes no environment)."""
    prelude = "".join(f"{k} = {v!r}\n" for k, v in globals_.items())
    return uecmd(prelude + Path(path).read_text())


def checked(output: str) -> str:
    """`output` of uecmd, or SystemExit when the editor side raised."""
    if "Traceback (most recent call last)" in output:
        raise SystemExit(f"the editor-side script failed:\n{output}")
    return output


def render_defaults_problems() -> list[str]:
    """Why this editor's pictures would not be the defined ones: a Nanite mesh in the level, Nanite shadows on,
    the editor viewport rendering. Empty when all is as the host project and the level builder set it.

    Nanite shadows cost 13 ms per capture however little moves and change the image at edges (TODO 12.7.4); the
    viewport shifts the captures' indirect light by ~2/255 (TODO 12.7.5). A URLab import builds Nanite meshes,
    so a level whose import skipped `in_editor/no_nanite.py` is caught here.
    """
    shadow, viewport, count, names = (checked(uecmd(_RENDER_DEFAULTS)).strip().split(" ", 3) + [""])[:4]
    problems = []
    if int(count):
        problems.append(
            f"{count} Nanite meshes ({names}): rebuild the level, bash scripts/build_scene.sh <scene> --unreal"
        )
    if int(shadow):
        problems.append(
            f"r.Shadow.Nanite {shadow}: the host project's DefaultEngine.ini sets 0 (scripts/setup_unreal.sh)"
        )
    if int(viewport):
        problems.append(f"ShowFlag.Rendering {viewport}: in_editor/cmd_server.py sets 0 when the editor starts")
    return problems


def require_render_defaults() -> None:
    problems = render_defaults_problems()
    if problems:
        raise SystemExit("[unreal] the editor is not in its render defaults: " + "; ".join(problems))


def connect(*, local_model: bool):
    """A URLab bridge client; with `local_model` it carries the editor's MuJoCo model (PIE must be running)."""
    from urlab_client import URLabClient

    client = URLabClient(URLAB_HOST, local_model=local_model, recv_timeout_ms=60000)
    try:
        client.connect()
    except ValueError:  # the handshake before PIE carries no model
        pass
    return client


def stop_pie() -> None:
    """End PIE if it runs. Before any client.close(): close() reverts URLab to live, and a live PIE has crashed
    the plugin's ZMQ publisher (UURLabZmqPublishTransport::PostStep, twice on 2026-09-29, TODO 12.17)."""
    client = connect(local_model=False)
    if client.manager_present:
        client.sim.stop()
        time.sleep(2.0)
    client.close()


def current_level() -> tuple[str, bool]:
    """(the editor world's level path, whether PIE runs)."""
    words = checked(uecmd(_LEVEL)).split()
    return (words[0], words[1] == "True") if len(words) == 2 else ("", words[-1:] == ["True"])


def load_level(level_path: str) -> None:
    """Open `level_path` in the editor (PIE stopped first); a no-op when it is open already."""
    path, pie = current_level()
    if path == level_path and not pie:
        return
    if pie:
        stop_pie()
    out = checked(
        uecmd(
            "les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)\n"
            f"print(unreal.EditorAssetLibrary.does_asset_exist({level_path!r}) and les.load_level({level_path!r}))"
        )
    )
    if out.strip() != "True":
        raise SystemExit(f"the editor could not open level {level_path}: {out.strip()}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m locomani.sim.unreal.editor '<python>' | -f <file.py>")
    code = Path(argv[1]).read_text() if argv[0] == "-f" else argv[0]
    print(uecmd(code), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
