"""Static regression checks for the Muse vLLM LoRA administration plugin."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = (ROOT / "scripts/h200/muse_file_endpoint.py").read_text()
START = (ROOT / "scripts/h200/start_muse_inference.sh").read_text()
TP_PATCH = (ROOT / "scripts/h200/muse_transformers_sitecustomize.py").read_text()
PROXY = (ROOT / "scripts/h200/muse_replica_proxy.py").read_text()
ORCHESTRATOR_START = (ROOT / "scripts/h200/start_orchestrator.sh").read_text()
ORCHESTRATOR_CONFIG = (ROOT / "configs/h200/orchestrator.muse15.toml").read_text()


class MuseLoRAEndpointTest(unittest.TestCase):
    def test_plugin_exposes_every_root_route_used_by_policy_updates(self):
        for route in ("/stage_lora_adapter", "/pause", "/resume", "/load_lora_adapter"):
            self.assertIn(f'@app.post("{route}")', ENDPOINT)

    def test_remote_adapter_is_staged_before_native_load(self):
        load = ENDPOINT.split('async def load_lora_adapter(request: Request):', 1)[1]
        self.assertLess(load.index("stage_lora_adapter_files"), load.index("handler.load_lora_adapter"))
        self.assertIn("lora_request.load_inplace = True", load)

    def test_engine_installs_plugin_and_enables_rank_128_lora(self):
        self.assertIn("VLLM_PLUGINS=muse_file_server", START)
        self.assertIn("--max-lora-rank 128", START)
        self.assertIn("--fully-sharded-loras", START)

    def test_engine_defaults_to_transformers_model_implementation(self):
        self.assertIn("model_impl=${MUSE_MODEL_IMPL:-transformers}", START)
        self.assertIn('--model-impl "$model_impl"', START)
        self.assertNotIn("--enforce-eager", START)
        self.assertIn("cudagraph_mode=${MUSE_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}", START)
        self.assertIn('\\"cudagraph_mode\\":\\"$cudagraph_mode\\"', START)

    def test_engine_defaults_to_eight_independent_tp1_replicas(self):
        self.assertIn("tp_size=${MUSE_TP_SIZE:-1}", START)
        self.assertIn("replica_count=$((8 / tp_size))", START)
        self.assertIn('--tensor-parallel-size "$tp_size"', START)
        self.assertIn("max_num_seqs=${MUSE_MAX_NUM_SEQS:-32}", START)
        self.assertIn('for replica in $(seq 0 $((replica_count - 1)))', START)
        self.assertIn('start_replica "$replica" "$devices" "$port"', START)
        self.assertIn("muse_replica_proxy.py", START)
        self.assertIn("sorted(rotated, key=lambda backend: _active[backend])", PROXY)

    def test_orchestrator_fans_adapter_updates_to_every_replica(self):
        for port in range(8001, 8009):
            self.assertIn(f"http://127.0.0.1:{port}/v1", ORCHESTRATOR_CONFIG)
        self.assertIn('config["model"]["client"]["admin_base_url"]', ORCHESTRATOR_START)

    def test_transformers_backend_uses_complete_attention_gate_tensors(self):
        self.assertIn("muse_transformers_sitecustomize.py", START)
        self.assertIn('("k_proj", "v_proj")', TP_PATCH)
        self.assertIn('self_attn.{projection}', TP_PATCH)
        self.assertIn('= "replicate"', TP_PATCH)
        self.assertIn('tp_size > 2', TP_PATCH)
        self.assertIn("MuseGlimmerModel._tp_plan", TP_PATCH)
        self.assertIn('f"language_model.{name}"', TP_PATCH)
        self.assertIn("isinstance(module, MuseGlimmerTextAttention)", TP_PATCH)
        self.assertIn("QKVFuser.match = classmethod", TP_PATCH)

    def test_transformers_backend_applies_muse_final_logit_transform(self):
        self.assertIn("TransformersMultiModalForCausalLM.compute_logits", TP_PATCH)
        self.assertIn("self.text_config.output_multiplier", TP_PATCH)
        self.assertIn("self.text_config.final_logit_softcapping", TP_PATCH)
        self.assertIn("torch.tanh(logits * multiplier / softcap) * softcap", TP_PATCH)


if __name__ == "__main__":
    unittest.main()
