MAX_BENCH_NUM = 10_000
MIN_BENCH_NUM = 0


class BenchMarkTotalError(Exception):
    def __init__(self, N):
        self.message = f"Total run num {N} is not supported, N should be in [{MIN_BENCH_NUM},{MAX_BENCH_NUM}]"
        super().__init__(self.message)


class WrongModeError(Exception):
    def __init__(self, mode):
        self.message = f"Mode {mode} is not supported"
        super().__init__(self.message)


class WrongDeviceError(Exception):
    def __init__(self, device):
        self.message = f"Device {device} is not supported"
        super().__init__(self.message)


class GpuIsNotAvailable(Exception):
    def __init__(self, message="Gpu is not available"):
        self.message = message
        super().__init__(self.message)


class TRTIsNotAvailable(Exception):
    def __init__(self, message="TRT is not available"):
        self.message = message
        super().__init__(self.message)
