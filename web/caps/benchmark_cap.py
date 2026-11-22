"""
Benchmark Cap: Evaluates CPU, GPU, and NPU/DirectML hardware inference performance.
Tests model throughput and latency across all available execution providers.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..core.box import SuperResolutionBox


class BenchmarkCap:
    """Adapter for running hardware inference benchmarks."""

    def __init__(self, box: SuperResolutionBox) -> None:
        self.box = box

    def run_benchmark(
        self, width: int = 320, height: int = 180, iterations: int = 40
    ) -> Dict[str, Any]:
        """Run latency & FPS benchmark across all detected providers."""
        if ort is None:
            return {"error": "onnxruntime is not installed"}

        original_provider = self.box.active_provider
        results: List[Dict[str, Any]] = []

        dummy_frame = np.random.randint(
            0, 255, (height, width, 3), dtype=np.uint8
        )

        test_configs = [
            ("CPU (Single-Thread - 1 Thread)", "CPUExecutionProvider", 1),
            ("CPU (Dual-Core - 2 Threads)", "CPUExecutionProvider", 2),
            ("CPU (Quad-Core - 4 Threads)", "CPUExecutionProvider", 4),
            ("CPU (Multi-Thread - 8 Threads)", "CPUExecutionProvider", 8),
        ]

        # Only add GPU benchmarks if genuinely present on physical hardware
        if self.box.has_cuda:
            test_configs.append(("NVIDIA GPU (CUDA)", "CUDAExecutionProvider", 1))
        if self.box.has_dml:
            test_configs.append(("DirectML (NPU / GPU)", "DmlExecutionProvider", 1))

        for label, provider, threads in test_configs:
            try:
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = threads
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                prov_list = list(dict.fromkeys([provider, "CPUExecutionProvider"]))
                sess = ort.InferenceSession(
                    str(self.box.model_path),
                    sess_options=sess_opts,
                    providers=prov_list,
                )
                active_p = sess.get_providers()[0]
                inp_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                expect_fp16 = "float16" in sess.get_inputs()[0].type

                # Prepare input tensor
                dtype = np.float16 if expect_fp16 else np.float32
                inp = (
                    np.random.rand(1, 1, height, width).astype(dtype)
                )

                # Warmup
                for _ in range(8):
                    sess.run([out_name], {inp_name: inp})

                # Timed iterations
                latencies = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    sess.run([out_name], {inp_name: inp})
                    latencies.append((time.perf_counter() - t0) * 1000.0)

                median_ms = float(np.median(latencies))
                min_ms = float(np.min(latencies))
                p95_ms = float(np.percentile(latencies, 95))
                fps = round(1000.0 / max(median_ms, 0.01), 1)

                results.append(
                    {
                        "target": label,
                        "provider": active_p,
                        "threads": threads,
                        "median_ms": round(median_ms, 2),
                        "min_ms": round(min_ms, 2),
                        "p95_ms": round(p95_ms, 2),
                        "fps": fps,
                        "success": True,
                    }
                )
            except Exception as err:
                results.append(
                    {
                        "target": label,
                        "provider": provider,
                        "threads": threads,
                        "error": str(err),
                        "success": False,
                    }
                )

        # Restore original provider
        self.box.switch_provider(original_provider)

        return {
            "model": self.box.model_path.name,
            "resolution": f"{width}×{height} -> {width*self.box.scale}×{height*self.box.scale}",
            "iterations": iterations,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
