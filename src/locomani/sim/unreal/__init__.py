"""Unreal (URLab) as a renderer of the MuJoCo state: the `images.source: unreal` path (TODO 12.9).

URLab (MuJoCo inside Unreal Engine 5.7, patched: patches/urlab_fresh_capture.md) runs in puppet mode: MuJoCo
physics, SONIC deploy, the policy server and lockstep all stay outside; a qpos goes in, the dataset cameras come
out. One long-lived editor per machine (`scripts/start_unreal.sh`) holds the scene's level; these modules run in
the URLab bridge venv ($BRIDGE_PY) and talk to it over URLab's bridge and the editor's command server.

    editor.py                  the running editor: Python in it, its level, its render defaults
    view.py                    UnrealView: qpos -> {stream: image}, the object the mirror and the re-render share
    mirror.py                  the eval mirror (locomani.sim.mirror's protocol)
    render_dataset_cameras.py  the dataset re-render (locomani.sim.view_render)
    export_mjcf.py             a scene's robot + carton as a self-contained MJCF for URLab's importer
    build_level.py             a scene's level: the room level plus that MJCF, materials, no Nanite
    in_editor/                 scripts that run inside the editor, sent by editor.run_file; not importable here
"""
