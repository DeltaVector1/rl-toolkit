import os
import sys
import unittest
from unittest.mock import patch

from prime_rl.trainer import nsys_worker


class NsightWorkerTest(unittest.TestCase):
    def test_non_target_worker_execs_normal_trainer(self):
        with patch.dict(
            os.environ,
            {"PRIME_CUDA_PROFILE": "1", "PRIME_CUDA_PROFILE_RANK": "24", "RANK": "25"},
            clear=True,
        ), patch.object(sys, "argv", ["nsys_worker", "@", "/tmp/config"]), patch.object(
            nsys_worker.os, "execv"
        ) as execv:
            nsys_worker.main()
        command = execv.call_args.args[1]
        self.assertEqual(command[1:4], ["-m", "prime_rl.trainer.rl.train", "@"])

    def test_target_worker_execs_one_cuda_nvtx_capture(self):
        env = {
            "PRIME_CUDA_PROFILE": "1",
            "PRIME_CUDA_PROFILE_RANK": "24",
            "RANK": "24",
            "PRIME_NSYS_BIN": "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys",
            "PRIME_NSYS_VERSION": "NVIDIA Nsight Systems version 2025.5.2",
            "PRIME_NSYS_OUTPUT": "/data/out/nsys-rank-24",
            "PRIME_NSYS_SESSION": "glmkernel-r24-test",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            sys, "argv", ["nsys_worker", "@", "/tmp/config"]
        ), patch.object(nsys_worker.Path, "is_file", return_value=True), patch.object(
            nsys_worker.os, "access", return_value=True
        ), patch.object(nsys_worker.os, "execv") as execv:
            nsys_worker.main()
        command = execv.call_args.args[1]
        self.assertEqual(command[1], "profile")
        self.assertIn("--trace=cuda,nvtx", command)
        self.assertIn("--capture-range-end=stop", command)
        self.assertEqual(command[-4:-1], ["-m", "prime_rl.trainer.rl.train", "@"])
        self.assertNotIn("PRIME_CUDA_PROFILE_SECONDS", " ".join(command))


if __name__ == "__main__":
    unittest.main()
