import sqlite3
import datetime


class MetadataStore:
    def __init__(self):
        connn = sqlite3.connect('metadata.db')
        cursor = connn.cursor()
        self.conn = connn
        self.cursor = cursor
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT,
            url TEXT,
            title TEXT,
            topic TEXT,
            credibility_score REAL DEFAULT NULL,
            faiss_position INTEGER DEFAULT NULL
        )
    ''')
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            topic TEXT PRIMARY KEY,
            frequency TEXT,
            last_run TEXT)
    ''')
        self.conn.commit()

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            content TEXT,
            created_at TEXT)''')
        self.conn.commit()

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            topic TEXT,
            status TEXT,
            result TEXT DEFAULT NULL,
            error TEXT DEFAULT NULL,
            created_at TEXT,
            finished_at TEXT DEFAULT NULL)''')
        self.conn.commit()

    def insert_chunk(self, chunk_id, text, url, title, topic, faiss_position):
        self.cursor.execute(
            '''
            INSERT INTO chunks (chunk_id, text, url, title, topic, faiss_position)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (chunk_id, text, url, title, topic, faiss_position)
        )
        self.conn.commit()

    def url_exists(self, url):
        self.cursor.execute(
            'SELECT 1 FROM chunks WHERE url = ? LIMIT 1',
            (url,)
        )
        row = self.cursor.fetchone()
        return row is not None

    def update_score(self, chunk_id: str, score: float):
        self.cursor.execute(
            "UPDATE chunks SET credibility_score = ? WHERE chunk_id = ?",
            (score, chunk_id)
        )
        self.conn.commit()

    def get_unscored_chunks(self, topic: str):
        self.cursor.execute(
            "SELECT chunk_id, text, url, title FROM chunks WHERE topic = ? AND credibility_score IS NULL",
            (topic,)
        )
        return self.cursor.fetchall()

    def get_chunk_by_position(self, faiss_position):
        self.cursor.execute(
            'SELECT chunk_id, text, url, title, credibility_score FROM chunks WHERE faiss_position = ?',
            (faiss_position,)
        )
        return self.cursor.fetchone()

    def add_subscription(self, topic, frequency):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute(
            '''
            INSERT OR REPLACE INTO subscriptions (topic, frequency, last_run)
            VALUES (?, ?, ?)
            ''',
            (topic, frequency, now)
        )
        self.conn.commit()

    def get_due_subscriptions(self):
        self.cursor.execute("SELECT topic, frequency, last_run FROM subscriptions")
        all_subs = self.cursor.fetchall()

        frequency_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30
        }
        now = datetime.datetime.now()
        due = []

        for topic, frequency, last_run in all_subs:
            last_run_date = datetime.datetime.strptime(last_run, "%Y-%m-%d %H:%M:%S")
            required_days = frequency_days.get(frequency, 1)
            days_passed = (now - last_run_date).days
            if days_passed >= required_days:
                due.append((topic, frequency))

        return due

    def get_subscription(self, topic):
        self.cursor.execute(
            "SELECT topic, frequency, last_run FROM subscriptions WHERE topic = ?",
            (topic,)
        )
        return self.cursor.fetchone()

    def get_chunk_by_id(self, chunk_id):
        self.cursor.execute(
            "SELECT chunk_id, text, url, title, credibility_score FROM chunks WHERE chunk_id = ?",
            (chunk_id,)
        )
        return self.cursor.fetchone()

    def save_digest(self, topic, content):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO digests (topic, content, created_at) VALUES (?, ?, ?)",
            (topic, content, now)
        )
        self.conn.commit()

    def get_latest_digest(self, topic):
        self.cursor.execute(
            "SELECT content, created_at FROM digests WHERE topic = ? ORDER BY id DESC LIMIT 1",
            (topic,)
        )
        return self.cursor.fetchone()


    def create_job(self, job_id, topic):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO jobs (job_id, topic, status, created_at) VALUES (?, ?, 'running', ?)",
            (job_id, topic, now)
        )
        self.conn.commit()

    def get_running_job(self, topic):
        self.cursor.execute(
            "SELECT job_id FROM jobs WHERE topic = ? AND status = 'running' LIMIT 1",
            (topic,)
        )
        row = self.cursor.fetchone()
        return row[0] if row else None

    def finish_job(self, job_id, status, result=None, error=None):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? WHERE job_id = ?",
            (status, result, error, now, job_id)
        )
        self.conn.commit()

    def get_job(self, job_id):
        self.cursor.execute(
            "SELECT job_id, topic, status, result, error, created_at, finished_at FROM jobs WHERE job_id = ?",
            (job_id,)
        )
        return self.cursor.fetchone()

    def fail_stale_jobs(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "UPDATE jobs SET status = 'failed', error = 'server restarted', finished_at = ? WHERE status = 'running'",
            (now,)
        )
        self.conn.commit()
        return self.cursor.rowcount

    def remove_subscription(self, topic):
        self.cursor.execute(
            "DELETE FROM subscriptions WHERE topic = ?",
            (topic,)
        )
        self.conn.commit()
        return self.cursor.rowcount