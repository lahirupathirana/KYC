"""
RQ worker entry point.

Run via: python -m app.workers.redis_worker
Or in Docker: the redis_worker service in docker-compose.yml uses this command.

Long-running inference tasks (e.g. Whisper on long audio, batch OCR)
are enqueued here so they don't block the FastAPI event loop or timeout
the HTTP response.
"""

import redis
from rq import Queue, Worker
from rq.job import Job

from app.core.config import settings

QUEUES = ["kyc_inference", "default"]


def get_queue(name: str = "kyc_inference") -> Queue:
    conn = redis.from_url(settings.redis_url)
    return Queue(name, connection=conn)


def enqueue_task(fn, *args, timeout: int = 300, **kwargs) -> str:
    """Enqueue a callable and return the job ID for polling."""
    job = get_queue().enqueue(fn, *args, job_timeout=timeout, **kwargs)
    return job.id


def get_job_status(job_id: str) -> dict:
    conn = redis.from_url(settings.redis_url)
    job = Job.fetch(job_id, connection=conn)
    return {
        "id": job.id,
        "status": job.get_status().value,
        "result": job.result,
        "error": str(job.exc_info) if job.exc_info else None,
    }


def run_worker() -> None:
    conn = redis.from_url(settings.redis_url)
    queues = [Queue(name, connection=conn) for name in QUEUES]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker()
