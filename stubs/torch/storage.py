from typing import Optional

class Storage:
    def cuda(self, cuda_device: Optional[int] = None) -> 'Storage': ...
