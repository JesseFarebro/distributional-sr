import fiddle as fdl


class Generator(fdl.Tag):
    """Generator related config."""


class Discriminator(fdl.Tag):
    """Discriminator related config."""


class DType(fdl.Tag):
    """DType."""


class LearningRate(fdl.Tag):
    """Learning rate."""


class OuterKernel(fdl.Tag):
    """The outer kernel."""


class OuterKernelBandwidth(fdl.Tag):
    """The bandwidth of the outer kernel."""


class InnerKernel(fdl.Tag):
    """The inner kernel."""


class InnerKernelBandwidth(fdl.Tag):
    """The bandwidth of the inner kernel."""
