from redis import Redis
from rq import Worker, Queue
import os

listen = ['monday']
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
conn = Redis.from_url(redis_url)

if __name__ == "__main__":
    queues = [Queue(name, connection=conn) for name in listen]
    worker = Worker(queues, connection=conn)
    print("🚀 Worker started. Waiting for jobs...")
    worker.work()
