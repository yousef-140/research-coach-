import threading

from agents.scout import run_scout
from agents.curator import run_curator
from agents.critic import run_critic
from digest import generate_digest

INGEST_LOCK = threading.Lock()  

def run_pipeline(topic: str):
    source_list = run_scout(topic)  

    with INGEST_LOCK:  
        chunks_stored = run_curator(source_list, topic)
        critic_result = run_critic(topic)

    return {
        "topic": topic,
        "chunks_stored": chunks_stored,
        "evaluations": critic_result.evaluations,
    }


if __name__ == "__main__":
    result = run_pipeline("creatine supplements")
    print(generate_digest(result))