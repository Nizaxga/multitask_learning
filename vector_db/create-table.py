import psycopg2
from config import HOST, DBNAME, USERNAME, PASSWORD

conn = psycopg2.connect(
    host=HOST,
    dbname=DBNAME,
    user=USERNAME,
    password=PASSWORD,
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE vector_datasets (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES vector_datasets(id) ON DELETE CASCADE,
    embedding VECTOR NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT now()
);

""")
conn.commit()
cursor.close()
conn.close()