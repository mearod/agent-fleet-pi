from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))
from judge.client import JudgeConfig  # noqa: E402
from judge.schema import scalar_result  # noqa: E402


class JudgeSchemaTest(unittest.TestCase):
    def test_normalizes_correctness_to_scalar_reward(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "nested" / "q1_eval.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"judge_result": {"correct": True}, "retrieval": {"recall": 1}}), encoding="utf-8")
            result = scalar_result(Path(root), "q1")
            self.assertEqual(result["reward"], 1.0)
            self.assertTrue(result["correct"])

    def test_remote_judge_reuses_fleet_gateway_without_double_v1(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "os.environ",
            {
                "BROWSECOMP_JUDGE_MODE": "openai",
                "BASE_URL": "https://gateway.example.invalid/v1",
                "MODEL": "fleet-model",
            },
            clear=True,
        ):
            root_path = Path(root)
            config = JudgeConfig.from_env(root_path, root_path / "gold", root_path / "eval")
            command = config.command(root_path / "runs")
            self.assertEqual(config.model, "fleet-model")
            self.assertIn("https://gateway.example.invalid/v1", command)
            self.assertNotIn("https://gateway.example.invalid/v1/v1", command)
            self.assertIn("--source-root", command)

    def test_remote_judge_normalizes_completion_endpoint_to_api_root(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "os.environ",
            {
                "BROWSECOMP_JUDGE_MODE": "openai",
                "BASE_URL": "https://gateway.example.invalid/v1/chat/completions",
                "MODEL": "fleet-model",
            },
            clear=True,
        ):
            root_path = Path(root)
            config = JudgeConfig.from_env(root_path, root_path / "gold", root_path / "eval")
            command = config.command(root_path / "runs")
            self.assertEqual(
                command[command.index("--base_url") + 1],
                "https://gateway.example.invalid/v1",
            )

    def test_remote_judge_bypasses_proxy_for_fleet_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "os.environ",
            {
                "BROWSECOMP_JUDGE_MODE": "openai",
                "BASE_URL": "https://gateway.example.invalid/v1",
                "MODEL": "fleet-model",
                "NO_PROXY": "existing.example",
            },
            clear=True,
        ), patch("judge.client.subprocess.run") as run:
            root_path = Path(root)
            config = JudgeConfig.from_env(root_path, root_path / "gold", root_path / "eval")
            config.evaluate(root_path / "runs")
            child_env = run.call_args.kwargs["env"]
            self.assertEqual(
                child_env["NO_PROXY"],
                "existing.example,gateway.example.invalid",
            )
            self.assertEqual(child_env["no_proxy"], child_env["NO_PROXY"])
            self.assertEqual(child_env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_local_judge_does_not_inherit_agent_model(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch.dict(
            "os.environ",
            {"BROWSECOMP_JUDGE_MODE": "local", "MODEL": "remote-agent-model"},
            clear=True,
        ):
            root_path = Path(root)
            config = JudgeConfig.from_env(root_path, root_path / "gold", root_path / "eval")
            self.assertEqual(config.model, "Qwen/Qwen3-32B")


if __name__ == "__main__":
    unittest.main()
