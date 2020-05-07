import torch

from allennlp.common.testing import AllenNlpTestCase, multi_device

actual_devices = set()


class TestTesting(AllenNlpTestCase):
    @multi_device
    def test_multi_device(self, device: str):
        actual_devices.add(device)

    def test_devices_accounted_for(self):
        expected_devices = {"cpu", "cuda"} if torch.cuda.is_available() else {"cpu"}
        assert expected_devices == actual_devices
