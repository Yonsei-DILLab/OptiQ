import unittest
from .entropy_followup import score


def records(a, b):
    return [dict(step=step, policy=dict(episodes=sum(c)+f, goals=c, failure=f))
            for step, c, f in ((800000, a, 0), (1000192, b, 0))]


class SelectionTest(unittest.TestCase):
    def test_rejects_collapsed_or_lucky_final(self):
        self.assertIsNone(score(records([200,0,0,0], [125]*4)))
        self.assertIsNone(score(records([50]*4, [490,4,3,3])))

    def test_success_alone_is_insufficient(self):
        self.assertIsNone(score(records([200,0,0,0], [500,0,0,0])))

    def test_low_success_is_rejected(self):
        r=records([50]*4, [100]*4)
        r[1]['policy'].update(episodes=500, failure=100)
        self.assertIsNone(score(r))

    def test_accepts_stable_four_goals(self):
        self.assertIsNotNone(score(records([50]*4, [125]*4)))
        self.assertIsNotNone(score(records([10,50,70,70], [25,125,175,175])))

    def test_requires_actual_final_step(self):
        self.assertIsNone(score(records([50]*4, [125]*4)[:1]))


if __name__ == '__main__': unittest.main()
