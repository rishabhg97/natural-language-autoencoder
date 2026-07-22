import asyncio
import importlib
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")


ROOT = pathlib.Path(__file__).resolve().parents[1]
NLA_ROOT = ROOT / "external" / "natural_language_autoencoders"
if str(NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(NLA_ROOT))


def _install_miles_stubs() -> None:
    modules = {
        "miles": types.ModuleType("miles"),
        "miles.rollout": types.ModuleType("miles.rollout"),
        "miles.rollout.generate_utils": types.ModuleType("miles.rollout.generate_utils"),
        "miles.rollout.generate_utils.generate_endpoint_utils": types.ModuleType(
            "miles.rollout.generate_utils.generate_endpoint_utils"
        ),
        "miles.rollout.inference_rollout": types.ModuleType("miles.rollout.inference_rollout"),
        "miles.rollout.inference_rollout.inference_rollout_train": types.ModuleType(
            "miles.rollout.inference_rollout.inference_rollout_train"
        ),
        "miles.utils": types.ModuleType("miles.utils"),
        "miles.utils.http_utils": types.ModuleType("miles.utils.http_utils"),
        "miles.utils.processing_utils": types.ModuleType("miles.utils.processing_utils"),
        "miles.utils.types": types.ModuleType("miles.utils.types"),
    }
    modules["miles.rollout.generate_utils.generate_endpoint_utils"].compute_request_payload = (
        lambda *args, **kwargs: ({}, None)
    )
    modules["miles.rollout.generate_utils.generate_endpoint_utils"].update_sample_from_response = (
        lambda *args, **kwargs: None
    )

    async def get_worker_urls(_args):
        raise AssertionError("router /workers should not be queried")

    async def post(_url, _payload):
        return {}

    modules["miles.rollout.inference_rollout.inference_rollout_train"].get_worker_urls = get_worker_urls
    modules["miles.utils.http_utils"].post = post
    modules["miles.utils.processing_utils"].load_tokenizer = lambda *args, **kwargs: None
    modules["miles.utils.types"].Sample = object
    sys.modules.update(modules)


class NLAGenerateUrlResolutionTests(unittest.TestCase):
    def setUp(self):
        _install_miles_stubs()
        self.mod = importlib.import_module("nla.rollout.nla_generate")
        self.mod._ENGINE_URLS = None

    def test_engine_urls_from_rollout_external_addrs_are_normalized(self):
        args = SimpleNamespace(rollout_external_engine_addrs=["127.0.0.1:31000", "http://host:31001"])

        self.assertEqual(
            self.mod._engine_urls_from_args(args),
            ["http://127.0.0.1:31000", "http://host:31001"],
        )

    def test_resolve_url_uses_external_engine_addrs_without_router_workers(self):
        args = SimpleNamespace(
            rollout_external_engine_addrs=["127.0.0.1:31000", "127.0.0.1:31001"],
            sglang_disable_radix_cache=True,
            sglang_router_ip="10.0.0.1",
            sglang_router_port=4188,
        )

        url = asyncio.run(self.mod._resolve_url(args, sample_index=3))

        self.assertEqual(url, "http://127.0.0.1:31001/generate")


if __name__ == "__main__":
    unittest.main()
