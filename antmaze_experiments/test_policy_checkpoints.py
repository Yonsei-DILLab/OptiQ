import unittest
from .policy_checkpoints import verify_regulator_state

class RegulatorCheckpointTest(unittest.TestCase):
    def test_off(self):
        verify_regulator_state({'regulator_enabled':False},{'regulator_enabled':False},False)
    def test_on(self):
        verify_regulator_state({'regulator':b'state'},{'regulator':b'state'},True)
    def test_missing_on(self):
        with self.assertRaises(AssertionError):verify_regulator_state({}, {}, True)
    def test_corrupted_on(self):
        with self.assertRaises(AssertionError):verify_regulator_state({'regulator':b'a'},{'regulator':b'b'},True)
    def test_off_must_be_explicit(self):
        with self.assertRaises(AssertionError):verify_regulator_state({}, {}, False)

if __name__=='__main__':unittest.main()
