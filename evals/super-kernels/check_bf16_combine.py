import os

import torch
import torch.distributed as dist


def counts_for_case(rank_count: int, name: str) -> list[list[int]]:
    if name == "balanced":
        return [[2] * rank_count for _ in range(rank_count)]
    if name == "uneven":
        return [[1 + ((src * 3 + dst) % 3) for dst in range(rank_count)] for src in range(rank_count)]
    if name == "zero":
        return [
            [0] * rank_count,
            *[[0] + [1 + ((src + dst) % 2) for dst in range(1, rank_count)] for src in range(1, rank_count)],
        ]
    raise ValueError(name)


def make_source(rows: int, columns: int, rank: int, requires_grad: bool = False):
    data = torch.empty((rows, columns), device="cuda", dtype=torch.bfloat16)
    if rows:
        values = torch.arange(rows * columns, device="cuda", dtype=torch.float32).reshape(rows, columns)
        data.copy_((values + rank * 0.25).to(torch.bfloat16))
    return data.requires_grad_(requires_grad)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    local_bad = int(actual.shape != expected.shape or not torch.equal(actual, expected))
    flag = torch.tensor([local_bad], device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise AssertionError(f"{label} differs on at least one rank")


def dispatch_owned(ep, source, output_splits, input_splits, group_name):
    return ep._BF16Dispatch.apply(
        source, output_splits, input_splits, group_name, "dispatch", None, "bf16"
    )


def run_case(ep, name: str, columns: int) -> None:
    rank = dist.get_rank()
    world = dist.get_world_size()
    matrix = counts_for_case(world, name)
    input_splits = matrix[rank]
    output_splits = [matrix[src][rank] for src in range(world)]
    rows = sum(input_splits)
    output_rows = sum(output_splits)
    group_name = dist.group.WORLD.group_name

    source = make_source(rows, columns, rank)
    dispatched = dispatch_owned(ep, source, output_splits, input_splits, group_name)
    expected = torch.empty((rows, columns), device="cuda", dtype=torch.bfloat16)
    dist.all_to_all_single(expected, dispatched, input_splits, output_splits)

    outputs = []
    for rounds in (8, 1):
        ep._BF16_COMBINE_ROUNDS = rounds
        value = dispatch_owned(ep, make_source(rows, columns, rank), output_splits, input_splits, group_name)
        outputs.append(ep._streaming_bf16_combine(value, input_splits, output_splits, group_name, rounds).clone())
    assert_equal(outputs[0], expected, f"{name} rounds=8")
    assert_equal(outputs[1], expected, f"{name} rounds=1")
    assert_equal(outputs[0], outputs[1], f"{name} round comparison")

    gradients = []
    weights = torch.arange(rows * columns, device="cuda", dtype=torch.float32).reshape(rows, columns)
    for rounds in (8, 1):
        ep._BF16_COMBINE_ROUNDS = rounds
        value = dispatch_owned(
            ep, make_source(rows, columns, rank, requires_grad=True), output_splits, input_splits, group_name
        )
        gradient_edge = torch.autograd.graph.get_gradient_edge(value)
        result = ep._BF16Dispatch.apply(
            value, input_splits, output_splits, group_name, "combine", None, "bf16"
        )
        gradient = torch.autograd.grad((result.float() * weights).sum(), gradient_edge)[0]
        expected_gradient = torch.empty((output_rows, columns), device="cuda", dtype=torch.bfloat16)
        dist.all_to_all_single(expected_gradient, weights.to(torch.bfloat16), output_splits, input_splits)
        assert_equal(gradient, expected_gradient, f"{name} rounds={rounds} inverse reference")
        gradients.append(gradient)
    assert_equal(gradients[0], gradients[1], f"{name} gradient comparison")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise RuntimeError(f"expected WORLD_SIZE=8, got {world}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        import prime_rl.trainer.distributed.expert_parallel as ep

        for name in ("balanced", "uneven", "zero"):
            run_case(ep, name, columns=17)
        dist.barrier()
        if dist.get_rank() == 0:
            print("BF16 combine rounds 8 and 1 match expected all-to-all outputs and gradients")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
