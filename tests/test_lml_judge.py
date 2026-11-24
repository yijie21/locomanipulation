"""TODO 49.4: the LocoManiLab judge (comply.py: did the robot do what the instruction says; loss.py: why a record is not a success).
The cases are the old beds' own selftests (comply_lml.py --selftest, loss_lml.py --selftest of the drawer bed, a superset of the
two others), run on the ported modules; plus the module's independence from the bed (numpy only)."""

import ast
import unittest
from pathlib import Path

from locomani.bench.locomanilab import comply, loss
from locomani.bench.locomanilab.comply import ART, ART_WHY, LIFT, NAV, PLACE, Watcher, _tl, verdict
from locomani.bench.locomanilab.loss import loss_of

HERE = Path(comply.__file__).parent


class Comply(unittest.TestCase):
    def test_old_selftest_cases(self):
        def v(fam, spec, success=True):
            return verdict(fam, success, _tl(spec).state())["why"]

        for fam in NAV + LIFT + PLACE + ART:
            assert v(fam, "R:3", success=False) == "not_success", fam
        # navigation: upright when the check holds (a fall earlier does not count: the robot may get up)
        assert v("navigation", "-:5 +-:2") == "" and v("navigation", "-:5 +!-:1") == "fell"
        assert v("navigation", "!-:4 +-:2") == ""
        for fam in LIFT:
            assert v(fam, "R:10 +R:1") == "" and v(fam, "R:5 +R:10") == "" and v(fam, "L:3 +R:10") == ""
            assert v(fam, "-:5 +R:2") == "short"  # 0.04 s in a hand
            assert v(fam, "R:9 +R:1") == "" and v(fam, "R:8 +R:1") == "short"
            assert v(fam, "R:10 RO:1 +R:5") == "short"  # a touch of the support breaks the run
            assert v(fam, "R:10 +RO:1") == "on_support" and v(fam, "R:10 +RC:1") == "on_support"
            assert v(fam, "R:10 +r:1") == "no_hand" and v(fam, "R:10 +-:3") == "no_hand"
            assert v(fam, "R:10 +R:1 +RO:2") == ""  # the run reached 0.2 s at an earlier check step
            assert v(fam, "Rr:10 +Rr:1") == ""  # the forearm may touch
        for fam in PLACE:
            assert v(fam, "O:3 R:5 Rr:1 -:3 +C:3") == ""
            assert v(fam, "R:5 -:2 +C:3") == "" and v(fam, "L:5 +O:1") == ""
            assert v(fam, "R:4 -:1 +C:3") == "not_carried"  # a flick, four steps only
            assert v(fam, "RO:10 O:3 +C:3") == "not_carried" and v(fam, "-:10 +O:3") == "not_carried"
            assert v(fam, "R:5 +RC:1") == "hand_on" and v(fam, "R:5 +L:1") == "hand_on"
            assert v(fam, "R:3 +RC:1") == "not_carried"  # both wrong: the first test reports
            assert v(fam, "RC:10 +C:1") == "not_carried"  # on the receptacle the whole time
            assert v(fam, "R:5 +C:1 R:5") == ""  # a hand after the check is not looked at
        for fam in ART[:2]:  # a door: upright and an attempt (the genuine flag is the caller's)
            w = lambda s: _tl(s).state()  # noqa: E731
            assert verdict(fam, True, w("-:5 +-:2")) == dict(ok=True, why="")
            assert verdict(fam, True, w("-:5 +-:2"), genuine=False)["why"] == ART_WHY[1]
            assert verdict(fam, True, w("-:5 +!-:1"))["why"] == ART_WHY[0]
            assert verdict(fam, True, w("-:5 +-:2"), genuine=False) != verdict(fam, True, w("-:5 +!-:1"), genuine=False)
            st = Watcher(
                strict=True
            )  # a fall before the first check-true step counts here (and not for the other families)
            st.feed(0.02, None, False, True)
            st.feed(0.04, None, True, False)
            assert verdict(fam, True, st.state())["why"] == ART_WHY[0]
        w = lambda s: _tl(s).state()  # noqa: E731
        # a drawer (hand_drives): the motion must come from a hand, not a leg; the arm and the torso may be in contact too
        for by, want in ((["hand"], ""), (["hand", "forearm", "torso"], ""), ([], ART_WHY[2]), (["forearm"], ART_WHY[2]),
                         (["knee"], ART_WHY[3]), (["hand", "shin"], ART_WHY[3]), (["foot"], ART_WHY[3])):  # fmt: skip
            assert verdict("articulation", True, w("-:5 +-:2"), moved_by=by, hand_drives=True)["why"] == want, by
        assert verdict("articulation", True, w("-:5 +-:2"), moved_by=[], hand_drives=False)[
            "ok"
        ]  # a door: no such rule
        assert verdict("articulation", True, w("-:5 +!-:1"), moved_by=["foot"], hand_drives=True)["why"] == ART_WHY[0]
        # the drawer pick-and-place: the drawer by a hand, then the place rule on the apple
        pnp = lambda spec, by=("hand",), **kw: verdict(
            "articulation_pnp", True, w(spec), moved_by=list(by), hand_drives=True, **kw
        )["why"]  # noqa: E731,E501
        assert pnp("O:3 R:5 Rr:1 -:3 +C:3") == "" and pnp("R:5 -:2 +C:3") == ""
        assert pnp("R:4 -:1 +C:3") == "not_carried" and pnp("-:10 +C:3") == "not_carried"
        assert pnp("R:5 +RC:1") == "hand_on" and pnp("R:5 +C:2", by=("shin",)) == ART_WHY[3]
        assert pnp("R:5 +C:2", by=()) == ART_WHY[2] and pnp("R:5 +!C:2") == ART_WHY[0]
        assert pnp("R:5 +C:2", genuine=False) == ART_WHY[1]
        try:
            verdict("push", True, {})
        except ValueError:
            pass
        else:
            raise AssertionError("unknown family")


class Loss(unittest.TestCase):
    def test_old_selftest_cases(self):
        held = dict(held=True, rise_cm=12.0, moved_cm=2.0, tilt_deg=5.0)
        drop = dict(held=False, rise_cm=0.0, moved_cm=2.0, tilt_deg=5.0)
        close = dict(held=True, rise_cm=0.0, moved_cm=1.0, tilt_deg=1.0)
        pp = dict(family="pick_place", success=False, end="limit", fell=None, found=True)

        def t(want, **kw):
            base = dict(pp)
            base.update(kw)
            assert loss_of(base) == want, (want, loss_of(base), kw)

        t("error", error="boom")
        t("error", end="error")
        t("error", success=True, genuine=False)
        t(None, success=True, end="success")
        t(None, success=True, comply=dict(ok=True, why=""))
        t("grasp", success=True, comply=dict(ok=False, why="not_carried"))
        t("place", success=True, comply=dict(ok=False, why="hand_on"))
        t("collide", success=True, comply=dict(ok=False, why="on_support"))
        t("fall", success=True, comply=dict(ok=False, why="fell"))
        t("fall", fell=3.2, end="limit")
        t("perceive", found=False)
        t("perceive", why="no target seen")
        t("navigate", family="navigation")
        t("navigate", family="navigation", end="limit", marks=dict(end=dict(held=False)))
        t("timeout", marks={})  # never reached the grasp, nothing wrong with the reaches
        t("timeout", marks=dict(before_close=close, after_lift=held, end=dict(held=True)))  # carrying at the limit
        t("deliver", end="done", marks={})
        t("deliver", why="no grasp", marks=dict(before_close=close))
        t("deliver", marks=dict(before_close=close, after_lift=drop), palm_err_cm=3.1)
        t("deliver", marks=dict(before_close=close, after_lift=drop), reach_ok=[True, False, True])
        t("collide", marks=dict(before_close=dict(close, moved_cm=4.0), after_lift=drop))
        t("collide", marks=dict(before_close=dict(close, tilt_deg=30.0), after_lift=drop))
        t("grasp", marks=dict(before_close=close, after_lift=drop), palm_err_cm=1.0, reach_ok=[True] * 3)
        t("grasp", family="lift", marks=dict(before_close=close, after_lift=held, end=dict(held=False)))
        t("timeout", family="lift", marks=dict(before_close=close, after_lift=held, end=dict(held=True)))
        t("grasp", marks=dict(before_close=close, after_lift=held, before_open=drop, end=drop))  # lost on the way
        t("collide", marks=dict(before_close=close, after_lift=held, before_open=dict(held, basket_cm=5.0), end=drop))
        t("deliver", marks=dict(before_close=close, after_lift=held, before_open=held, end=drop), release_cm=3.0)
        t("place", marks=dict(before_close=close, after_lift=held, before_open=held, end=drop), release_cm=1.0)
        t("deliver", marks=dict(before_close=close, after_lift=held, end=dict(held=True)), end="done")
        # doors: error, fall, perceive, then art_not_reached, art_no_move, art_partial, art_not_held, timeout
        art = dict(frac_start=0.1, frac_max=0.4, frac_min=0.1, t_goal=5.0, hold_best_s=0.4, hand_min_cm=2.0, touch=["hand"],
                   goal="fraction >= 0.50 held 1.0 s")  # fmt: skip
        door = dict(family="articulation", art=art, found=True)
        t("error", error="boom", **door)
        t("fall", fell=2.0, **door)
        t("perceive", **dict(door, found=False))
        t("art_not_reached", **dict(door, art=dict(art, hand_min_cm=9.0, touch=[])))
        t("art_no_move", **dict(door, art=dict(art, frac_max=0.12, touch=["hand"])))  # a touch, no move
        t(
            "art_no_move", **dict(door, art=dict(art, hand_min_cm=9.0, touch=["arm"], frac_max=0.1))
        )  # touched with the arm
        t("art_partial", **dict(door, art=dict(art, t_goal=None)))
        t("art_not_held", **door)
        t("timeout", **dict(door, art=dict(art, hold_best_s=1.0)))
        t(
            "timeout",
            **dict(
                door, family="navigation_articulation", art=dict(art, goal="fraction >= 0.40 and robot within 0.75 m")
            ),
        )
        t("art_not_reached", **dict(door, art=dict(art, touch=[], hand_min_cm=9.0)))  # first match wins
        t(None, **dict(door, success=True, comply=dict(ok=True, why="")))
        t("fall", **dict(door, success=True, comply=dict(ok=False, why="fell before the door check held")))
        t("error", **dict(door, success=True, genuine=False, comply=dict(ok=False, why="not genuine (too few steps)")))
        # drawers (bed3): the finer touch names, the hand-drives rejections
        t("art_no_move", **dict(door, art=dict(art, hand_min_cm=9.0, touch=["forearm"], frac_max=0.1)))
        t(
            "art_not_reached", **dict(door, art=dict(art, hand_min_cm=9.0, touch=["knee", "foot"]))
        )  # a leg on it is no reach
        t("art_not_reached", **dict(door, success=True, comply=dict(ok=False, why="drawer moved by a leg")))
        t("art_not_reached", **dict(door, success=True, comply=dict(ok=False, why="drawer not moved by a hand")))
        # the drawer pick-and-place: the apple's pick, then the drawer's stages, then the place
        pa = dict(frac_start=0.2, frac_max=0.85, frac_min=0.2, frac_end=0.2, t_goal=9.0, hold_best_s=0.4, hand_min_cm=2.0,
                  touch=["hand"], goal="fraction >= 0.80 held 0.1 s, then apple inside and fraction <= 0.20 held 1.0 s")  # fmt: skip
        pnp = dict(family="articulation_pnp", art=pa, found=True, success=False, end="limit", fell=None)
        ms = lambda **k: dict(before_close=close, after_lift=held, before_open=held, end=drop) | k  # noqa: E731
        t("error", **dict(pnp, error="boom"))
        t("fall", **dict(pnp, fell=2.0))
        t("perceive", **dict(pnp, found=False))
        t("grasp", **dict(pnp, marks=dict(before_close=close, after_lift=drop), palm_err_cm=1.0))  # the pick came first
        t("collide", **dict(pnp, marks=dict(before_close=dict(close, moved_cm=4.0), after_lift=drop)))
        t("deliver", **dict(pnp, why="no grasp"))
        t(
            "art_not_reached", **dict(pnp, art=dict(pa, frac_max=0.2, hand_min_cm=9.0, touch=[]), marks={})
        )  # never opened
        t("art_no_move", **dict(pnp, art=dict(pa, frac_max=0.21), marks={}))
        t("art_partial", **dict(pnp, art=dict(pa, frac_max=0.5, t_goal=None), marks={}))
        t("timeout", **dict(pnp, marks={}))  # opened, no grasp tried, the limit
        t("deliver", **dict(pnp, marks={}, end="done"))
        t("grasp", **dict(pnp, marks=ms(before_open=drop)))  # lost on the way
        t("deliver", **dict(pnp, marks=ms(before_open=held), release_cm=3.0))
        t("art_partial", **dict(pnp, art=dict(pa, t_goal=None, frac_end=0.6), marks=ms(), release_cm=1.0))  # left open
        t("place", **dict(pnp, art=dict(pa, t_goal=None), marks=ms(), release_cm=1.0))  # closed, apple not inside
        t("art_not_held", **dict(pnp, marks=ms(), release_cm=1.0))
        t("timeout", **dict(pnp, art=dict(pa, hold_best_s=1.0), marks=ms(), release_cm=1.0))
        t(None, **dict(pnp, success=True, comply=dict(ok=True, why="")))
        t("grasp", **dict(pnp, success=True, comply=dict(ok=False, why="not_carried")))
        t("place", **dict(pnp, success=True, comply=dict(ok=False, why="hand_on")))


class Independent(unittest.TestCase):
    def test_numpy_only(self):
        for f in ("comply.py", "loss.py"):
            tree = ast.parse((HERE / f).read_text())
            mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertLessEqual(mods, {"numpy", "re", "__future__"}, f)

    def test_art_families_agree(self):
        self.assertEqual(tuple(comply.ART), tuple(loss.ART))


if __name__ == "__main__":
    unittest.main()
