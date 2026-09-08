# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : ray_inference_utils.py

import asyncio
import logging
from typing import Callable, List

import ray

logger = logging.getLogger(__name__)


@ray.remote
class _RayActorSchedulerActor:
    """Ray-side scheduler.

    - Driver-side `RayActorScheduler.submit()` returns an ObjectRef immediately
      by calling this actor method.
    - This actor method waits for an available actor token, dispatches the real
      actor task via `actor_fn`, awaits its completion, and then returns the
      actor token.

    This keeps user experience identical to plain Ray actor calls:
        ref = scheduler.submit(x)
        out = ray.get(ref)
    while avoiding driver-side blocking in `submit()`.
    """

    def __init__(
        self,
        actors: List[ray.actor.ActorHandle],
        actor_fn: Callable[[ray.actor.ActorHandle, ...], ray.ObjectRef],
        queue_max_size: int,
    ):
        self.actors = actors
        self.actor_fn = actor_fn
        self.queue_max_size = queue_max_size

        self._actor_queue: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()
        for _ in range(queue_max_size):
            for a in actors:
                self._actor_queue.put_nowait(a)

    async def submit(self, *args, **kwargs):
        actor = await self._actor_queue.get()
        try:
            obj_ref = self.actor_fn(actor, *args, **kwargs)
            if not isinstance(obj_ref, ray.ObjectRef):
                raise TypeError(
                    "The actor_fn should return a ray.ObjectRef, "
                    f"but got {type(obj_ref)}"
                )
            return await obj_ref
        finally:
            # Always return the token to keep load balancing working.
            self._actor_queue.put_nowait(actor)


class RayActorScheduler:
    """
    A high-performance load balanced scheduler for Ray actors.
    Balance the load of multiple Ray actors using token bucket.

    Args:
        actors: A list of Ray actor handles to schedule work onto.
        actor_fn: A callable that takes an actor handle plus user-provided
            `*args/**kwargs` and submits a Ray task (typically an actor method
            call). It **must** return a `ray.ObjectRef` representing the
            submitted task.
        queue_max_size: Token bucket size per actor. Effectively caps the
            number of in-flight tasks allowed per actor to this value.
        scheduler_max_concurrency: Concurrency for internal Ray scheduler actor.
    """

    def __init__(
        self,
        actors: List[ray.actor.ActorHandle],
        actor_fn: Callable[[ray.actor.ActorHandle, ...], ray.ObjectRef],
        queue_max_size: int = 2,
    ):
        assert actors and all(isinstance(a, ray.actor.ActorHandle) for a in actors), (
            "actors must be a non-empty list of Ray actor handles"
        )
        assert queue_max_size > 0 and isinstance(queue_max_size, int), (
            "queue_max_size must be a positive integer"
        )

        # A single Ray actor does both dispatching and monitoring.
        # submit() on driver returns immediately.
        self._scheduler_actor = _RayActorSchedulerActor.options(
            max_concurrency=queue_max_size
            * len(actors)
            * 2  # Allow some extra concurrency for waiting
        ).remote(actors=actors, actor_fn=actor_fn, queue_max_size=queue_max_size)

    def submit(self, *args, **kwargs):
        return self._scheduler_actor.submit.remote(*args, **kwargs)


class _RayActorMethodProxy:
    """Proxy one actor method through a RayActorScheduler."""

    def __init__(self, scheduler: RayActorScheduler, method_name: str):
        self._scheduler = scheduler
        self._method_name = method_name

    def remote(self, *args, **kwargs):
        """Submit this method using the original Ray actor call style."""
        return self._scheduler.submit(self._method_name, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Alias for ``remote`` for more convenient scheduler usage."""
        return self.remote(*args, **kwargs)


class RayActorSchedulerWrapper:
    """Actor-like wrapper that load balances method calls across Ray actors.

    The wrapper keeps the normal Ray actor method style while using
    RayActorScheduler internally:

        wrapper = RayActorSchedulerWrapper(actors)
        ref = wrapper.inference.remote(x)

    Direct calls are also supported as an alias:

        ref = wrapper.inference(x)

    Args:
        actors: A list of Ray actor handles of the same interface.
        queue_max_size: Token bucket size per actor. Effectively caps the
            number of in-flight tasks allowed per actor to this value.
    """

    def __init__(
        self,
        actors: List[ray.actor.ActorHandle],
        queue_max_size: int = 2,
    ):
        self._scheduler = RayActorScheduler(
            actors=actors,
            actor_fn=lambda actor, method_name, *args, **kwargs: getattr(
                actor, method_name
            ).remote(*args, **kwargs),
            queue_max_size=queue_max_size,
        )

    def __getattr__(self, method_name: str) -> _RayActorMethodProxy:
        if method_name.startswith("_"):
            raise AttributeError(method_name)
        return _RayActorMethodProxy(self._scheduler, method_name)


__all__ = ["RayActorScheduler", "RayActorSchedulerWrapper"]
