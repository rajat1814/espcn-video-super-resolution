"""Caps (Adapters) wrapping the pure Super-Resolution Box."""
from .photo_cap import PhotoCap
from .video_cap import VideoCap
from .live_cap import LiveCap
from .benchmark_cap import BenchmarkCap

__all__ = ["PhotoCap", "VideoCap", "LiveCap", "BenchmarkCap"]
