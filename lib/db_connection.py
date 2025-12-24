import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def test_connection():
    conn = get_db_connection()
    if conn:
        print("Successfully connected to the database!")
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"PostgreSQL version: {version['version']}")
        cur.close()
        conn.close()
    else:
        print("Failed to connect.")


if __name__ == "__main__":
    test_connection()
