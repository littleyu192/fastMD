import unittest

from fastmd.execution.cuda_graph import _smallest_fitting_key


class CudaGraphCapacityTest(unittest.TestCase):
    def test_smallest_fitting_key_keeps_a_padding_pair(self):
        # A cached bucket fits only with at least one padding undirected pair
        # (key[0] > u); among the fitting keys the smallest is reused.
        cases = {
            "reuses the smallest larger cached capacity": (
                {(37376, 942080), (37376, 950272), (37888, 942080), (37888, 950272)},
                37000, 933000, (37376, 942080)),
            "rejects a full undirected capacity": ({(37376, 950272)}, 37376, 940000, None),
        }
        for name, (keys, u, t, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_smallest_fitting_key(keys, u, t, 512, 8192), expected)


if __name__ == "__main__":
    unittest.main()
