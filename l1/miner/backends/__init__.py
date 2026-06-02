"""GPU inference backends for the InferenceToken miner."""
from .detect import get_backend, get_available_backends
from .base import InferenceBackend, MockBackend
from .jacobi_backend import LookaheadBackend, JacobiBackend  # JacobiBackend is an alias
