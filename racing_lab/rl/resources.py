"""Portable process accounting; missing measurements remain null, never zero."""

from collections import defaultdict
from contextlib import contextmanager
import os
import platform
import time


def process_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return dict(
                rss_bytes=counters.WorkingSetSize, peak_rss_bytes=counters.PeakWorkingSetSize
            )
    else:
        import resource

        scale = 1 if platform.system() == "Darwin" else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        current = None
        try:
            from pathlib import Path

            current = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
        except (OSError, ValueError, IndexError):
            pass
        return dict(rss_bytes=current, peak_rss_bytes=peak)
    return dict(rss_bytes=None, peak_rss_bytes=None)


def hardware():
    cpu = platform.processor() or None
    ram = None
    if os.name == "nt":
        import ctypes
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual",
                    "virtual_available",
                    "extended",
                )
            ]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            ram = memory.total
    else:
        try:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            pass
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), physical_memory_bytes=ram)


class ResourceMeter:
    def __init__(self, device="cpu"):
        self.device = device
        self.wall_start, self.cpu_start = time.perf_counter(), time.process_time()
        self.seconds = defaultdict(float)
        self.before = process_memory()

    def _synchronize(self):
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, phase):
        self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.seconds[phase] += time.perf_counter() - started

    def report(self):
        wall = time.perf_counter() - self.wall_start
        result = dict(
            wall_seconds=wall,
            process_cpu_seconds=time.process_time() - self.cpu_start,
            component_seconds=dict(self.seconds),
            unassigned_seconds=max(0, wall - sum(self.seconds.values())),
            memory_at_start=self.before,
            memory=process_memory(),
            memory_scope="process lifetime peak; includes Python/framework startup",
            gpu_peak_allocated_bytes=None,
        )
        if str(self.device).startswith("cuda"):
            import torch

            result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
        return result
