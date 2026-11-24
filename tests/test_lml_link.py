"""TODO 49.4: the bed's link to the environment server (client.ServerLink) with a fake server file: start-up, the arguments it is
given (the benchmark's root and the seed floor), the retry of a failed reset, a refusal that is not retried, a dead server dropped
and restarted, a server that never comes up."""

import os
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from locomani.bench.locomanilab import client
from locomani.bench.locomanilab.config import LocoManiLabConfig

FAKE = textwrap.dedent(
    """
    import sys, os, json
    from multiprocessing.connection import Listener
    sock, out, root, floor, radii = sys.argv[1:6]
    open(os.path.join(out, "args.json"), "w").write(json.dumps(sys.argv[1:]))
    lis = Listener(sock, family="AF_UNIX")
    n = 0
    while True:
        c = lis.accept()
        while True:
            try:
                m = c.recv()
            except EOFError:
                break
            if m[0] == "quit":
                c.send({"ok": True}); sys.exit(0)
            if m[0] == "die":
                os._exit(3)
            if m[0] == "reset":
                n += 1
                if m[3] + m[2] < int(floor):
                    c.send({"error": "PermissionError: seed wall"})
                elif n == 1:
                    c.send({"error": "ParseError: unclosed token"})
                else:
                    c.send({"ok": True, "tries": n})
            else:
                c.send({"echo": list(m)})
    """
)


class Link(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        t = self.t = Path(self.td.name)
        (t / "fake_server.py").write_text(FAKE)
        import sys

        self.cfg = LocoManiLabConfig(
            lml_python=sys.executable,
            server_file=str(t / "fake_server.py"),
            lml_root=str(t),
            sock_dir=str(t),
            server_ready_s=10,
        )
        self.link = client.ServerLink(self.cfg, t / "srv")

    def tearDown(self):
        self.link.reset_link()
        self.td.cleanup()

    def test_the_server_gets_its_arguments_and_answers(self):
        self.assertEqual(self.link.ask("hello", 1), {"echo": ["hello", 1]})
        args = __import__("json").loads((self.t / "srv" / "args.json").read_text())
        self.assertEqual(args[1:], [str(self.t / "srv"), str(self.t), "100000", "1"])
        self.assertTrue(args[0].startswith(str(self.t / "lml_")))

    def test_a_failed_reset_is_tried_again_but_a_refusal_is_not(self):
        import unittest.mock as mock

        with mock.patch.object(time, "sleep"):
            rep = self.link.ask("reset", "T", 0, 100000, True, 0.05, 0.02)
        self.assertEqual(rep["tries"], 2)
        with self.assertRaises(AssertionError) as e:
            self.link.ask("reset", "T", 5, 3, True)  # the wall: seed 3 + episode 5 is under the floor
        self.assertIn("PermissionError", str(e.exception))

    def test_a_dead_server_is_dropped_and_the_next_ask_starts_a_new_one(self):
        self.link.ask("hello")
        first = self.link.proc.pid
        try:
            self.link.ask("die")
        except EOFError:
            pass
        time.sleep(0.3)
        self.assertTrue(self.link.dead())
        self.link.reset_link()
        self.assertFalse(self.link.dead())
        self.assertEqual(self.link.ask("again"), {"echo": ["again"]})
        self.assertNotEqual(self.link.proc.pid, first)

    def test_a_server_that_never_comes_up_is_an_error_not_a_hang(self):
        (self.t / "fake_server.py").write_text("import sys\nsys.exit(1)\n")
        with self.assertRaises(RuntimeError):
            self.link.ask("x")
        self.assertIsNone(self.link.conn)

    def test_quit_ends_the_server(self):
        self.link.ask("hello")
        pid = self.link.proc.pid
        self.link.ask("quit")
        self.link.proc.wait(10)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
