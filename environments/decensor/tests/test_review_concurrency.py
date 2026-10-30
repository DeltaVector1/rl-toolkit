import asyncio
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decensor_v1 import agentic_review, service, taskset
from decensor_v1.state import DecensorState


def test_reward_reviews_overlap_across_rollouts(monkeypatch):
    async def check():
        active = Counter()
        peaks = Counter()
        four_started = asyncio.Event()
        release = asyncio.Event()

        async def review(config, source):
            episode = source.episode
            active[episode] += 1
            peaks[episode] = max(peaks[episode], active[episode])
            if sum(active.values()) == 4:
                four_started.set()
            try:
                await release.wait()
            finally:
                active[episode] -= 1

        async def request(name, operation, **kwargs):
            return await operation()

        monkeypatch.setattr(taskset, "run_agentic_review", review)
        monkeypatch.setattr(taskset, "request_judge_score", request)
        monkeypatch.setattr(taskset, "find_model_response_failure", lambda *a, **k: (None, {}))
        monkeypatch.setattr(
            taskset, "build_turn_review_sources",
            lambda prompt, completion, **kwargs: [
                SimpleNamespace(episode=completion[0], turns=[SimpleNamespace(turn=i)])
                for i in range(4)
            ],
        )
        jobs = []
        for episode in range(2):
            task = SimpleNamespace(
                config=taskset.DecensorTaskConfig(),
                data=SimpleNamespace(
                    word_requirement="", word_target=0, response_format="plaintext",
                    harm_type="unknown", source_row_index=0, source_row_repeated=False,
                    has_style=False, system_prompt="",
                ),
                _split_conversation=lambda trace, episode=episode: ([], [episode]),
            )
            trace = SimpleNamespace(info={}, state=DecensorState(), is_truncated=False)
            jobs.append(asyncio.create_task(taskset.DecensorTask.decensor_reward(task, trace)))
        try:
            await asyncio.wait_for(four_started.wait(), timeout=2)
            await asyncio.sleep(0)
            assert active == {0: 2, 1: 2}
            assert peaks == {0: 2, 1: 2}
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        assert sum(active.values()) == 0

    asyncio.run(check())


def test_shared_judge_request_limit_remains_24(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_judge_slot_directory", tmp_path)

    async def check():
        config = SimpleNamespace(max_concurrent_requests=24)
        full = asyncio.Event()
        release = asyncio.Event()
        active = 0
        peak = 0
        entered = 0

        async def request():
            nonlocal active, peak, entered
            async with service.judge_request_slot(config):
                active += 1
                entered += 1
                peak = max(peak, active)
                if active == 24:
                    full.set()
                try:
                    await release.wait()
                finally:
                    active -= 1

        jobs = [asyncio.create_task(request()) for _ in range(25)]
        try:
            await asyncio.wait_for(full.wait(), timeout=2)
            await asyncio.sleep(0.03)
            assert active == entered == peak == 24
            release.set()
            await asyncio.wait_for(asyncio.gather(*jobs), timeout=2)
            assert entered == 25 and active == 0 and peak == 24
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

    try:
        asyncio.run(check())
    finally:
        service._close_inherited_capacity_owner()


def test_rollouts_share_judge_transport_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_judge_slot_directory", tmp_path)

    async def check():
        reviews = Counter()
        requests = Counter()
        four_reviews = asyncio.Event()
        full = asyncio.Event()
        release = asyncio.Event()
        peak = 0
        entered = 0

        async def create(*, episode):
            nonlocal peak, entered
            requests[episode] += 1
            entered += 1
            peak = max(peak, sum(requests.values()))
            if sum(requests.values()) == 3:
                full.set()
            try:
                await release.wait()
            finally:
                requests[episode] -= 1

        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ))

        async def review(config, source):
            reviews[source.episode] += 1
            if sum(reviews.values()) == 4:
                four_reviews.set()
            try:
                await agentic_review._judge_completion(
                    client, config, episode=source.episode,
                )
            finally:
                reviews[source.episode] -= 1

        monkeypatch.setattr(taskset, "run_agentic_review", review)
        monkeypatch.setattr(taskset, "find_model_response_failure", lambda *a, **k: (None, {}))
        monkeypatch.setattr(
            taskset, "build_turn_review_sources",
            lambda prompt, completion, **kwargs: [
                SimpleNamespace(episode=completion[0], turns=[SimpleNamespace(turn=i)])
                for i in range(4)
            ],
        )
        jobs = []
        for episode in range(2):
            config = taskset.DecensorTaskConfig()
            config.judge.max_concurrent_requests = 3
            task = SimpleNamespace(
                config=config,
                data=SimpleNamespace(
                    word_requirement="", word_target=0, response_format="plaintext",
                    harm_type="unknown", source_row_index=0, source_row_repeated=False,
                    has_style=False, system_prompt="",
                ),
                _split_conversation=lambda trace, episode=episode: ([], [episode]),
            )
            trace = SimpleNamespace(info={}, state=DecensorState(), is_truncated=False)
            jobs.append(asyncio.create_task(taskset.DecensorTask.decensor_reward(task, trace)))
        try:
            await asyncio.wait_for(four_reviews.wait(), timeout=2)
            await asyncio.wait_for(full.wait(), timeout=2)
            await asyncio.sleep(0.03)
            assert reviews == {0: 2, 1: 2}
            assert all(requests[episode] > 0 for episode in range(2))
            assert sum(requests.values()) == entered == peak == 3
        finally:
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        assert sum(reviews.values()) == sum(requests.values()) == 0
        # Cancellation must release the shared capacity for the next request.
        release.set()
        await asyncio.wait_for(
            agentic_review._judge_completion(client, config.judge, episode=2), timeout=2,
        )
        assert entered == 4 and peak == 3

    try:
        asyncio.run(check())
    finally:
        service._close_inherited_capacity_owner()
