"""TODO 49.5b: the judge on fake outcomes: the benchmark's own verdict needs no hand contact, so the hands must have moved the object
and no other link may have pressed on it."""

import unittest

from locomani.bench.humanoidarena import judge as J

HAND = "left_hand_palm_link"
HAND2 = "right_hand_thumb_1_link"
KNEE = "left_knee_link"
FOREARM = "left_wrist_yaw_link"
TORSO = "torso_link"


def outcome(contacts, success=True, fell=False, move=0.8):
    return dict(success=success, fell=fell, contacts=contacts, steps=600, obj_move=move)


def steps(a, b):
    return list(range(a, b))


class Hands(unittest.TestCase):
    def test_only_the_hands_links_are_hands(self):
        for link in (HAND, HAND2, "left_hand_camera_base_link", "right_hand_index_0_link"):
            self.assertTrue(J.is_hand(link), link)
        for link in (KNEE, FOREARM, TORSO, "pelvis", "left_ankle_roll_link", "left_wrist_roll_link"):
            self.assertFalse(J.is_hand(link), link)


class Official(unittest.TestCase):
    def test_success_and_no_fall(self):
        self.assertTrue(J.official(outcome({})))
        self.assertFalse(J.official(outcome({}, success=False)))
        self.assertFalse(J.official(outcome({}, fell=True)))


class Pp(unittest.TestCase):
    def v(self, contacts, **kw):
        return J.verdict("pp_box", outcome(contacts, **kw))

    def test_hands_carried_it(self):
        r = self.v({HAND: steps(100, 400), HAND2: steps(100, 400)})
        self.assertTrue(r["ok"])
        self.assertEqual((r["hand_steps"], r["body_steps"], r["why"]), (300, 0, ""))

    def test_the_benchmark_passes_but_no_hand_touched_it(self):
        r = self.v({})
        self.assertTrue(r["official"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["why"], "no_hand")

    def test_a_torso_pushed_it_even_with_hands_on_it(self):
        r = self.v({HAND: steps(100, 200), TORSO: steps(150, 160)})
        self.assertEqual((r["ok"], r["why"], r["body_parts"]), (False, "body_push", [TORSO]))

    def test_a_graze_is_allowed(self):
        self.assertTrue(self.v({HAND: steps(100, 200), KNEE: [150, 151]})["ok"])

    def test_a_forearm_is_the_body_not_a_hand(self):
        self.assertEqual(self.v({FOREARM: steps(0, 50)})["why"], "no_hand")

    def test_steps_are_counted_once_each(self):
        r = self.v({HAND: [1, 2, 3], HAND2: [2, 3, 4], KNEE: [4]})
        self.assertEqual((r["hand_steps"], r["body_steps"]), (4, 1))

    def test_an_object_that_hardly_moved_is_not_a_carry(self):
        r = self.v({HAND: steps(0, 50)}, move=0.02)
        self.assertEqual((r["ok"], r["why"]), (False, "not_moved"))

    def test_not_success_and_fell(self):
        self.assertEqual(self.v({HAND: steps(0, 50)}, success=False)["why"], "not_success")
        self.assertEqual(self.v({HAND: steps(0, 50)}, fell=True)["why"], "fell")


class NoRecord(unittest.TestCase):
    def test_open_door_has_no_object_record(self):
        r = J.verdict("open_door", outcome({}))
        self.assertEqual((r["official"], r["ok"], r["why"]), (True, False, "no_record"))


if __name__ == "__main__":
    unittest.main()
