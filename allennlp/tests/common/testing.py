import torch

from allennlp.common.testing import AllenNlpTestCase, multi_device


class TestTesting(AllenNlpTestCase):
    def test_multi_device(self):
        actual_devices = set()

        @multi_device
        def dummy_func(_self, device: str):
            # Have `self` as in class test functions.
            nonlocal actual_devices
            actual_devices.add(device)

        dummy_func(self)

        expected_devices = {"cpu", "cuda"} if torch.cuda.is_available() else {"cpu"}
        self.assertSetEqual(expected_devices, actual_devices)
