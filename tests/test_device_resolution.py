from __future__ import annotations

import unittest

import torch

from dataset.models.providers.transformers.hf_base_runner import HFBaseRunner


class TestDeviceResolution(unittest.TestCase):
    def test_auto_and_cpu(self) -> None:
        auto_device = HFBaseRunner._resolve_device("auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(auto_device.type, expected)

        self.assertEqual(HFBaseRunner._resolve_device("cpu").type, "cpu")

    def test_cuda_index_resolution(self) -> None:
        if not torch.cuda.is_available():
            self.assertEqual(HFBaseRunner._resolve_device("cuda:1").type, "cpu")
            return

        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.assertEqual(str(HFBaseRunner._resolve_device("cuda:1")), "cuda:1")
        else:
            with self.assertRaises(ValueError):
                HFBaseRunner._resolve_device("cuda:1")

    def test_invalid_device_raises(self) -> None:
        with self.assertRaises(ValueError):
            HFBaseRunner._resolve_device("not_a_real_device")


if __name__ == "__main__":
    unittest.main()
