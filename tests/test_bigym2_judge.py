"""TODO 49.5: the judge on fake outcomes: BiGym's own verdict is not enough, the hands must have done it and no body link may push."""

import unittest

from locomani.bench.bigym2 import judge as J

HAND = "g1_29dof_with_dex1_1/left_dex1_finger_link_1"
HAND2 = "g1_29dof_with_dex1_1/right_dex1_base_link"
KNEE = "g1_29dof_with_dex1_1/left_knee_link"
FOREARM = "g1_29dof_with_dex1_1/left_wrist_yaw_link"
DRAWER = "base_cabinet_600/drawer_small_4"


def outcome(contacts, success=True, fell=False):
    return dict(success=success, fell=fell, contacts=contacts, steps=600)


def steps(a, b):
    return list(range(a, b))


class Hands(unittest.TestCase):
    def test_only_the_grippers_own_bodies_are_hands(self):
        # the model prefix `g1_29dof_with_dex1_1/` names the gripper too: it once made every robot body a hand
        self.assertTrue(J.is_hand(HAND))
        self.assertTrue(J.is_hand(HAND2))
        for body in (KNEE, FOREARM, "g1_29dof_with_dex1_1/pelvis", "g1_29dof_with_dex1_1/left_ankle_roll_link"):
            self.assertFalse(J.is_hand(body), body)


class Official(unittest.TestCase):
    def test_success_and_no_fall(self):
        self.assertTrue(J.official(outcome({})))
        self.assertFalse(J.official(outcome({}, success=False)))
        self.assertFalse(J.official(outcome({}, fell=True)))


class Contact(unittest.TestCase):
    def v(self, contacts, **kw):
        return J.verdict("drawer_top_open", outcome(contacts, **kw))

    def test_hand_pulled_it(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(100, 200)})
        self.assertTrue(r["ok"])
        self.assertEqual((r["hand_steps"], r["body_steps"], r["why"]), (100, 0, ""))

    def test_the_benchmark_passes_but_no_hand_touched_it(self):
        r = self.v({})
        self.assertTrue(r["official"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["why"], "no_hand")

    def test_a_knee_pushed_it_even_with_a_hand_on_it(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(100, 200), f"{KNEE}|{DRAWER}": steps(150, 160)})
        self.assertFalse(r["ok"])
        self.assertEqual((r["why"], r["body_parts"]), ("body_push", ["left_knee_link"]))

    def test_a_forearm_is_the_body_not_a_hand(self):
        r = self.v({f"{FOREARM}|{DRAWER}": steps(0, 50)})
        self.assertEqual(r["why"], "no_hand")
        self.assertEqual(r["body_steps"], 50)

    def test_a_graze_of_the_body_is_allowed(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(100, 200), f"{KNEE}|{DRAWER}": steps(100, 102)})
        self.assertTrue(r["ok"])

    def test_too_short_a_touch_is_no_hand(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(100, 100 + J.HAND_STEPS - 1)})
        self.assertEqual(r["why"], "no_hand")

    def test_steps_are_counted_once_across_two_hands(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(0, 10), f"{HAND2}|{DRAWER}": steps(5, 15)})
        self.assertEqual(r["hand_steps"], 15)

    def test_fell_is_a_failure_even_when_the_hand_did_it(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(0, 50)}, fell=True)
        self.assertEqual((r["ok"], r["why"], r["official"]), (False, "fell", False))

    def test_no_success_is_not_success(self):
        r = self.v({f"{HAND}|{DRAWER}": steps(0, 50)}, success=False)
        self.assertEqual((r["ok"], r["why"]), (False, "not_success"))


class Reach(unittest.TestCase):
    def test_reach_needs_no_object_contact(self):
        self.assertTrue(J.verdict("reach_target_single", outcome({}))["ok"])

    def test_reach_still_fails_on_a_fall_or_no_success(self):
        self.assertEqual(J.verdict("reach_target_single", outcome({}, fell=True))["why"], "fell")
        self.assertEqual(J.verdict("reach_target_single", outcome({}, success=False))["why"], "not_success")


if __name__ == "__main__":
    unittest.main()
