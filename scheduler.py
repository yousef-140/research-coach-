from apscheduler.schedulers.background import BackgroundScheduler
import time
import logging
from tools.metadata_store import MetadataStore
from pipeline import run_pipeline
from digest import build_and_save_digest


def check_and_run_subscriptions():
    metadata_store = MetadataStore()
    due = metadata_store.get_due_subscriptions()
    for topic, frequency in due:
        try:
            logging.info(f"Running subscription for topic: {topic} with frequency: {frequency}")
            result = run_pipeline(topic)
            build_and_save_digest(result)
            metadata_store.add_subscription(topic, frequency)
        except Exception as e:
            logging.error(f"Subscription run failed for topic {topic}: {type(e).__name__}: {e}")
        time.sleep(15)


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_run_subscriptions, 'interval', hours=1)
    scheduler.start()
    logging.info("Scheduler started. Checking for due subscriptions every hour")