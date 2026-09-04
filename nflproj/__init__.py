"""NFL projection model.

The only thing that happens at import is one allocator setting, explained in
``_tune_allocator``.
"""
from __future__ import annotations

import ctypes
import logging
import platform

log = logging.getLogger(__name__)

# glibc's ``mallopt`` parameter number for the mmap threshold.
_M_MMAP_THRESHOLD = -3
# Simulated stat arrays are 20,000 float32 values - 78 KB each - and a slate
# holds several thousand of them.
_MMAP_THRESHOLD_BYTES = 64 * 1024


def _tune_allocator() -> bool:
    """Make simulation arrays go to mmap rather than the heap.

    This is not micro-optimisation; it is the difference between an app that
    survives being used and one that is killed.

    A simulated statistic is one array per player per stat: 20,000 values. At
    float64 that is 156 KB, which is above glibc's default 128 KB mmap
    threshold, so each array gets its own mapping and the memory goes straight
    back to the operating system when the slate is dropped. Storing samples as
    float32 - which halved the resting footprint - took them to 78 KB, *below*
    that threshold, so several thousand of them per slate came off the heap
    instead. Heap memory can only be returned when the freed chunk sits at the
    top of the heap, and interleaved with pandas objects it never does. The
    result was a process whose live object count was correctly bounded at two
    slates while its resident size climbed 626 -> 870 -> 1205 -> 1517 MB as a
    reader browsed weeks, and was killed on a 1 GB container.

    Lowering the threshold to 64 KB puts those arrays back on mmap. Setting it
    explicitly also pins it: left alone, glibc *raises* the threshold
    dynamically whenever it sees an mmap'd block freed, which would reintroduce
    the same failure after enough churn.

    Returns whether the call was made; a non-glibc platform simply keeps its
    own allocator behaviour.
    """
    if platform.system() != "Linux":
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD_BYTES) == 1
    except Exception as exc:          # musl, a hardened libc, anything unusual
        log.debug("could not tune allocator (%s); using platform defaults", exc)
        return False


ALLOCATOR_TUNED = _tune_allocator()
