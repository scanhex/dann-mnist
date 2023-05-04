import unittest
import itertools
from collections import Counter
import numpy as np

from ada.datasets.sampler import (
    BalancedBatchSampler,
    ReweightedBatchSampler,
)


class TestBatchSamplers(unittest.TestCase):
    class MyDataset:
        def __init__(self):
            self.targets = np.tile(["1", "2", "3", "3", "3", "4"], 10)

    @staticmethod
    def _idx_to_class(idx):
        idx = idx % 6
        if idx < 2:
            return str(idx + 1)
        elif idx < 5:
            return "3"
        else:
            return "4"

    def test_balanced_small_batches(self):
        sampler = BalancedBatchSampler(self.MyDataset(), batch_size=4 * 4)
        self.assertEqual(len(sampler), 3)
        batches = list(itertools.islice(sampler, 8))
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len(batch), 16)
            counts = Counter(map(self._idx_to_class, batch))
            self.assertEqual(counts, {"1": 4, "2": 4, "3": 4, "4": 4})

        # no duplicates in the 2 first batches
        first_batches = batches[0] + batches[1]
        self.assertEqual(len(first_batches), len(set(first_batches)))

        # duplicates for classes 1, 2, 4 in the last batch
        for i in range(2, 3):
            first_batches += batches[i]
        counts = Counter(first_batches)
        self.assertEqual(
            {self._idx_to_class(idx) for idx, nb in counts.items() if nb > 1},
            {"1", "2", "4"},
        )

    def test_balanced_bigger_batches(self):
        sampler = BalancedBatchSampler(self.MyDataset(), batch_size=12 * 4)
        self.assertEqual(len(sampler), 1)
        batches = list(itertools.islice(sampler, 3))
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch), 48)
        counts = Counter(map(self._idx_to_class, batch))
        self.assertEqual(counts, {"1": 12, "2": 12, "3": 12, "4": 12})

        # 2 duplicates for class 1, 2, 4
        counts = Counter(batch)
        self.assertEqual(
            sorted(self._idx_to_class(idx) for idx, nb in counts.items() if nb > 1),
            ["1", "1", "2", "2", "4", "4"],
        )

    def test_reweighter(self):
        sampler = ReweightedBatchSampler(
            self.MyDataset(), batch_size=11, class_weights=np.array([1, 2, 1, 3])
        )
        self.assertEqual(len(sampler), 5)
        one_part = 1 / (1 + 2 + 1 + 3)
        batches = list(itertools.islice(sampler, 10))
        self.assertEqual(len(batches), 5)
        counts = Counter()
        for _ in range(1000):
            for batch in sampler:
                self.assertEqual(len(batch), 11)
                counts.update(map(self._idx_to_class, batch))
        self.assertEqual(len(counts), 4)
        total = sum(counts.values())
        self.assertEqual(total, 1000 * 5 * 11)
        self.assertAlmostEqual(counts["1"] / total, one_part, 2)
        self.assertAlmostEqual(counts["2"] / total, 2 * one_part, 2)
        self.assertAlmostEqual(counts["3"] / total, one_part, 2)
        self.assertAlmostEqual(counts["4"] / total, 3 * one_part, 2)
