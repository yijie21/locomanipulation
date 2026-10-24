"""Vision RSI L3: the perception services' protocol (locomani.tasks.scene.services) -- a request answered over the
local connection, a server error raised at the caller, masks packed for the wire, and the category phrase an object
is prompted by."""

import os
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import services as S  # noqa: E402

PORT = 18850 + os.getpid() % 40  # clear of the real services' 18821 / 18822


def handle(req: dict) -> dict:
    if req.get("fail"):
        raise ValueError("asked to fail")
    return dict(echo=req["x"] * 2, mask=S.pack_mask(req["mask"]))


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threading.Thread(target=S.serve, args=(("127.0.0.1", PORT), handle, lambda *_: None), daemon=True).start()
        time.sleep(0.3)

    def test_a_request_is_answered_and_an_error_raised(self):
        c = S.Client(("127.0.0.1", PORT))
        mask = np.random.default_rng(0).random((48, 64)) > 0.5
        reply = c.call(dict(x=np.arange(3), mask=mask))
        np.testing.assert_array_equal(reply["echo"], [0, 2, 4])
        np.testing.assert_array_equal(S.unpack_mask(reply["mask"]), mask)
        with self.assertRaisesRegex(RuntimeError, "asked to fail"):
            c.call(dict(fail=True))
        self.assertEqual(c.call(dict(x=np.ones(1), mask=mask))["echo"][0], 2)  # still serving after the error

    def test_two_clients_are_served(self):
        a, b = S.Client(("127.0.0.1", PORT)), S.Client(("127.0.0.1", PORT))
        m = np.zeros((2, 2), bool)
        self.assertEqual((a.call(dict(x=1, mask=m))["echo"], b.call(dict(x=2, mask=m))["echo"]), (2, 4))


class Prompts(unittest.TestCase):
    def test_a_phrase_becomes_its_category(self):
        cases = {
            "the orange": "orange", "the coffee can": "coffee can", "the white mug": "white mug",
            "the TV remote": "TV remote", "the first pattern-2 candle": "candle", "the second small vase": "small vase",
            "the tallest potted plant": "potted plant", "the wolf figurine": "wolf figurine",
        }  # fmt: skip
        for phrase, want in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(S.prompt_of({"semantic": {"phrase": phrase}}, "X"), want)

    def test_without_a_phrase_the_key_is_split(self):
        self.assertEqual(S.prompt_of({}, "CandlePattern02_1"), "Candle")
        self.assertEqual(S.prompt_of({}, "WoodenSpoon"), "Wooden Spoon")
        self.assertEqual(S.prompt_of({}, "NIT_CoffeeCan_Anon"), "NIT Coffee Can")


if __name__ == "__main__":
    unittest.main()
