# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_qps_control.py

import asyncio
import time

from pytest import approx, mark, param

from dl_utils import QPSLimiter


def _fn(sleep_time, return_value):
    time.sleep(sleep_time)
    return return_value


@mark.parametrize(
    "max_qps, test_queries, sleep_time",
    [
        param(0.5, 1, 0.1),
        param(1, 1, 0.1),
        param(20, 5, 0.02),
    ],
)
def test_qps_limiter(max_qps, test_queries, sleep_time):
    return_value = 123456

    async def main():
        qps_limiter = QPSLimiter(max_qps=max_qps)

        start_time = time.time()
        results = await asyncio.gather(
            *[
                qps_limiter.run(_fn, sleep_time, return_value=return_value)
                for _ in range(test_queries)
            ]
        )
        duration = time.time() - start_time
        expected_duration = test_queries / max_qps + sleep_time

        assert all(x == return_value for x in results)
        assert duration == approx(expected_duration, abs=0.2)
        assert qps_limiter.real_qps() > 0

    asyncio.run(main())


def test_qps_limiter_runs_many_small_tasks():
    async def main():
        qps_limiter = QPSLimiter(max_qps=100)
        results = await asyncio.gather(
            *[qps_limiter.run(_fn, 0.001, return_value=i) for i in range(20)]
        )

        assert sorted(results) == list(range(20))
        assert qps_limiter.real_qps() > 0

    asyncio.run(main())
