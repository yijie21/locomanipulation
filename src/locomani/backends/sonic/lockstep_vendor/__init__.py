"""Vendored from SIMPLE (MIT, (c) 2025-2026 The SIMPLE Authors), commit
948022a: the Python side of the SONIC simulation-clock lockstep protocol.
Copied rather than imported so this repo does not track a peer working tree.
Only the import path in lockstep_unitree_bridge.py was changed. Lived at src/teleop_free/lockstep_vendor, a top-level
package, until 2026-09-27: after the move to src/locomani nothing put that directory on the path, so a lockstep
sim died at import (`No module named 'lockstep_vendor'`) and only a test that prepended it still passed.
"""
