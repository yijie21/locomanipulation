import unittest

from locomani.pipeline.eval_closed_loop import round_outcomes


class RoundOutcomeTests(unittest.TestCase):
    def attempt(self, rnd, attempt, ok, why=""):
        return dict(case_id=f"6000:{rnd}", round=rnd, attempt=attempt, ok=ok, why=why)

    def rate(self, rows):
        ended = round_outcomes(rows)
        return sum(1 for r in ended if r["ok"]), len(ended)

    def test_one_attempt_per_round_is_unchanged(self):
        rows = [self.attempt(0, 0, True), self.attempt(1, 0, False), self.attempt(2, 0, True)]
        self.assertEqual(self.rate(rows), (2, 3))

    def test_a_round_that_succeeds_on_the_retry_counts_once_and_as_a_success(self):
        # The case this exists for: the old reader called this round 1/2 = 50%.
        rows = [
            self.attempt(0, 0, False, "[stand] robot did not stabilise after reset"),
            self.attempt(0, 1, True),
        ]
        self.assertEqual(self.rate(rows), (1, 1))

    def test_a_round_that_fails_every_attempt_counts_once_and_as_a_failure(self):
        rows = [
            self.attempt(0, 0, False, "[stand] robot did not stabilise after reset"),
            self.attempt(0, 1, False, "[stand] robot did not stabilise after reset"),
        ]
        self.assertEqual(self.rate(rows), (0, 1))

    def test_the_round_reports_the_attempt_it_ended_on(self):
        rows = [self.attempt(3, 0, False, "[stand] no"), self.attempt(3, 1, True)]
        (ended,) = round_outcomes(rows)
        self.assertEqual((ended["attempt"], ended["ok"]), (1, True))

    def test_rounds_keep_first_seen_order(self):
        rows = [self.attempt(0, 0, True), self.attempt(1, 0, False), self.attempt(1, 1, True)]
        self.assertEqual([r["round"] for r in round_outcomes(rows)], [0, 1])

    def test_a_manifest_without_the_round_field_is_one_round_per_line(self):
        rows = [dict(ok=True), dict(ok=False), dict(ok=True)]
        self.assertEqual(self.rate(rows), (2, 3))


if __name__ == "__main__":
    unittest.main()
