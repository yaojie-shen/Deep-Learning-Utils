# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Common-Utils
# @File    : test_ray_inference_utils.py

import os
import time
from collections import Counter

import pytest

ray = pytest.importorskip("ray")
pytestmark = pytest.mark.skipif(
    os.environ.get("COMMON_UTILS_RUN_RAY_TESTS") != "1",
    reason="Ray actor integration tests are opt-in because they start local Ray workers.",
)


from common_utils import RayActorScheduler, RayActorSchedulerWrapper  # noqa: E402


@ray.remote
class Worker:
    def __init__(self, idx: int, sleep_time: float):
        self.idx = idx
        self.sleep_time = sleep_time

    def process(self, x):
        time.sleep(self.sleep_time)
        return x, self.idx


def test_scheduler_balances_load():
    """One slow actor should receive only one task when queue_max_size=1."""
    ray.init(ignore_reinit_error=True)

    try:
        # Create fast actors and one slow actor, then wait for them to be ready.
        actors = [Worker.remote(idx, 0.01) for idx in range(3)]
        actors.append(Worker.remote(3, 0.1))

        # actor_fn: call the remote process method
        def actor_fn(actor, x):
            return actor.process.remote(x)

        scheduler = RayActorScheduler(actors, actor_fn, queue_max_size=1)
        refs = []

        # wait for all actors to be ready before testing submit time
        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(scheduler._scheduler_actor.__ray_ready__.remote())

        # submit task
        start_submit = time.time()
        for i in range(13):
            refs.append(scheduler.submit(i))
        submit_time = time.time() - start_submit

        # wait for all tasks to finish
        start_wait = time.time()
        schdule_count = Counter()
        for i, ref in enumerate(refs):
            x, worker_idx = ray.get(ref)
            assert x == i, f"Expected {i} but got {x}"
            schdule_count[worker_idx] += 1
        wait_time = time.time() - start_wait

        assert submit_time < 0.5, f"Submit should be fast, but took {submit_time:.2f}s"
        assert wait_time < 1, f"Tasks should finish quickly, but took {wait_time:.2f}s"
        assert schdule_count[3] == 1, (
            f"Slow worker should process 1 task, but processed {schdule_count[3]}"
        )
    finally:
        ray.shutdown()


def test_scheduler_wrapper_supports_actor_like_remote_calls():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(idx, 0.0) for idx in range(2)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        refs = [wrapper.process.remote(i) for i in range(8)]
        results = ray.get(refs)

        assert [x for x, _ in results] == list(range(8))
        assert {worker_idx for _, worker_idx in results}.issubset({0, 1})
    finally:
        ray.shutdown()


def test_scheduler_wrapper_supports_direct_call_alias():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(idx, 0.0) for idx in range(2)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        refs = [wrapper.process(i) for i in range(8)]
        results = ray.get(refs)

        assert [x for x, _ in results] == list(range(8))
        assert {worker_idx for _, worker_idx in results}.issubset({0, 1})
    finally:
        ray.shutdown()


def test_scheduler_wrapper_resolves_missing_methods_lazily():
    ray.init(ignore_reinit_error=True)

    try:
        actors = [Worker.remote(0, 0.0)]
        wrapper = RayActorSchedulerWrapper(actors, queue_max_size=1)

        ray.get([a.__ray_ready__.remote() for a in actors])
        ray.get(wrapper._scheduler._scheduler_actor.__ray_ready__.remote())

        ref = wrapper.missing_method.remote(1)

        assert isinstance(ref, ray.ObjectRef)
        with pytest.raises(Exception):
            ray.get(ref)
    finally:
        ray.shutdown()
