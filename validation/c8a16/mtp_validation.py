"""Small real-model MTP gate; diagnostic RPC hooks are not performance instrumentation."""

import argparse
import json
import math
from pathlib import Path


class C8A16ValidationWorker:
    def install_mtp_probe(self):
        import torch_npu

        from vllm_ascend.attention.attention_c8a16 import AscendC8A16MetadataBuilder

        runner = self.model_runner
        self.mtp_probe = {
            "mode5_calls": 0,
            "invalid_mode5_calls": 0,
            "metadata": [],
            "postprocess": [],
            "gdn_postprocess_calls": 0,
        }
        probe = self.mtp_probe
        original_fia = torch_npu.npu_fused_infer_attention_score

        def fia(*args, **kwargs):
            if kwargs.get("key_antiquant_mode") == 5:
                probe["mode5_calls"] += 1
                q = args[0] if args else kwargs["query"]
                if (
                    q.shape[1] != 1
                    or kwargs.get("value_antiquant_mode") != 5
                    or kwargs.get("sparse_mode") != 0
                    or kwargs.get("atten_mask") is not None
                ):
                    probe["invalid_mode5_calls"] += 1
            return original_fia(*args, **kwargs)

        torch_npu.npu_fused_infer_attention_score = fia
        original_build = AscendC8A16MetadataBuilder.build

        def build(builder, *args, **kwargs):
            result = original_build(builder, *args, **kwargs)
            if len(probe["metadata"]) < 128:
                probe["metadata"].append(
                    {
                        "q_ends": result.actual_seq_lengths_q,
                        "kv_lens": result.seq_lens_list.copy(),
                        "decode_requests": result.c8a16_decode_requests,
                    }
                )
            return result

        AscendC8A16MetadataBuilder.build = build
        original_postprocess = runner.postprocess_sampled

        def postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc=None):
            # Deliberately synchronizing, small correctness runs only. This
            # observes the normal sampler and does not force acceptance.
            if len(probe["postprocess"]) < 512:
                probe["postprocess"].append(
                    {
                        "sampled": num_sampled.cpu().tolist(),
                        "rejected": num_rejected.cpu().tolist() if num_rejected is not None else [0] * len(num_sampled),
                    }
                )
            return original_postprocess(idx_mapping, sampled_tokens, num_sampled, num_rejected, query_start_loc)

        runner.postprocess_sampled = postprocess
        state = runner.model_state
        if "Mamba" in type(state).__name__:
            original_state = state.postprocess_state

            def postprocess_state(*args, **kwargs):
                probe["gdn_postprocess_calls"] += 1
                return original_state(*args, **kwargs)

            state.postprocess_state = postprocess_state
        return type(state).__name__

    def read_mtp_probe(self):
        return self.mtp_probe


def counters(llm):
    result = {}
    for metric in llm.get_metrics():
        if "spec_decode_num_" in metric.name and hasattr(metric, "value"):
            result[metric.name] = result.get(metric.name, 0) + metric.value
    return result


def main():
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="int8")
    parser.add_argument("--mtp", choices=("on", "off"), default="on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    options = dict(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.80,
        block_size=128,
        seed=2026,
        # vLLM's LLM() defaults disable_log_stats=True, which makes
        # get_metrics() assert 'Stat logging disabled'. The spec-decode
        # counters read below come from /metrics, so they must be enabled.
        disable_log_stats=False,
        language_model_only=True,
        worker_extension_cls="validation.c8a16.mtp_validation.C8A16ValidationWorker",
    )
    if args.mtp == "on":
        options["speculative_config"] = {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": True}
    llm = LLM(**options)
    states = llm.collective_rpc("install_mtp_probe")
    tokenizer = llm.get_tokenizer()
    prompts = []
    for index, length in enumerate((127, 128, 129, 4000)):
        tokens = tokenizer.encode(
            f"Request {index}: explain why prime numbers are infinite. ", add_special_tokens=False
        )
        prompts.append({"prompt_token_ids": (tokens * (length // len(tokens) + 1))[:length]})
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, logprobs=5) for n in (65, 97, 129, 193)]
    before = counters(llm)
    records = []
    # Different output lengths retire slots at different times. Run again to
    # exercise request-slot reuse and recurrent-state initialization.
    for repeat in range(2):
        for request, sampling in zip(llm.generate(prompts, params, use_tqdm=False), params):
            output = request.outputs[0]
            assert len(output.token_ids) == sampling.max_tokens
            assert output.finish_reason == "length"
            assert all(math.isfinite(p.logprob) for row in output.logprobs for p in row.values())
            records.append(
                {
                    "repeat": repeat,
                    "prompt_tokens": len(request.prompt_token_ids),
                    "token_ids": list(output.token_ids),
                    "top_logprobs": [
                        {str(token): value.logprob for token, value in row.items()} for row in output.logprobs
                    ],
                    "finish_reason": output.finish_reason,
                }
            )
    after = counters(llm)
    delta = {key: value - before.get(key, 0) for key, value in after.items()}
    probes = llm.collective_rpc("read_mtp_probe")
    assert all(p["invalid_mode5_calls"] == 0 for p in probes)
    if args.kv_cache_dtype == "int8":
        assert all(p["mode5_calls"] > 0 for p in probes), "new operator was not called"
    drafts = delta.get("vllm:spec_decode_num_drafts", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
    if args.mtp == "on":
        assert drafts > 0 and accepted > 0, "MTP did not actually propose/accept tokens"
        assert all(p["gdn_postprocess_calls"] > 0 for p in probes), "GDN postprocess not observed"
    coverage = {"all_accepted": 0, "partial_accepted": 0, "all_rejected": 0}
    for row in probes[0]["postprocess"]:
        for sampled, rejected in zip(row["sampled"], row["rejected"]):
            if sampled > 1 and rejected == 0:
                coverage["all_accepted"] += 1
            elif sampled > 1 and rejected > 0:
                coverage["partial_accepted"] += 1
            elif sampled == 1 and rejected > 0:
                coverage["all_rejected"] += 1
    result = {
        "functional_status": "pass",
        "kv_cache_dtype": args.kv_cache_dtype,
        "mtp": args.mtp,
        "states": states,
        "metrics_delta": delta,
        "acceptance_length": 1 + accepted / drafts if drafts else None,
        "observed_sampler_coverage": coverage,
        "outputs": records,
        "probes": probes,
        "limitation": "hooks and counters do not prove recurrent-state numerical equivalence; inspect unobserved cases",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("outputs", "probes")}))


if __name__ == "__main__":
    main()
