import os
import sqlite3

def setup_database(DB_FILE):
    """Sets up the SQLite DB file if it doesn't exist."""
    if os.path.exists(DB_FILE):
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    create_tables_sql = [
        """CREATE TABLE ICM_EPI_Columns (
            DEpsdNumber INTEGER PRIMARY KEY,
            StorEpsdType TEXT, DiagEpsdType TEXT, Specialness BOOLEAN,
            VT_EpisodeDuration REAL, VT_AverageHeartRate REAL, VT_MaxHeartRate REAL
        );""",
        """CREATE TABLE ICM_RSP_Columns (
            DEpsdNumberEpisode INTEGER PRIMARY KEY,
            EpsdStartSyncTS TEXT, EpsdEndSyncTS TEXT, StreamDataType INTEGER,
            TimeOfFirstSample TEXT, ExciterCurrent INTEGER,
            FOREIGN KEY(DEpsdNumberEpisode) REFERENCES ICM_EPI_Columns(DEpsdNumber)
        );""",
        """CREATE TABLE ICM_RSP_DATA_Columns (
            DEpsdNumber INTEGER, Obsolescence INTEGER, MarkerSelector INTEGER,
            FOREIGN KEY(DEpsdNumber) REFERENCES ICM_EPI_Columns(DEpsdNumber)
        );"""
    ]
    for sql in create_tables_sql: cursor.execute(sql)

    data_ICM_EPI_Columns = [
        (101, 14, 0, 1, 15.5, 120.5, 180.0), (102, 5, 2, 0, 10.0, 111.0, 120.0),
        (103, 1, 7, 0, 5.0, 140.0, 155.0), (104, 14, 0, 0, 22.5, 125.0, 190.0),
        (105, 5, 2, 0, 12.0, 110.0, 115.0), (106, 1, 7, 1, 8.0, 145.0, 160.0),
        (107, 14, 0, 0, 18.0, 118.0, 175.0)
    ]
    data_ICM_RSP_Columns = [
        (101, '2024-10-09 10:00:00', '2024-10-09 10:00:15', 4, '2024-10-09 10:00:00', 1),
        (102, '2023-01-08 11:01:00', '2023-01-08 11:01:30', 2, '2023-01-08 11:01:00', 3),
        (103, '2024-04-10 11:02:00', '2024-04-10 11:02:22', 3, '2024-04-10 11:02:00', 3),
        (104, '2024-02-12 21:10:00', '2024-02-12 21:10:21', 5, '2024-02-12 21:10:00', 2),
        (105, '2024-02-21 22:22:00', '2024-02-21 22:22:12', 4, '2024-02-21 22:22:00', 3),
        (106, '2024-09-03 08:19:00', '2024-09-03 08:19:34', 4, '2024-09-03 08:19:00', 2),
        (107, '2024-11-19 19:00:00', '2024-11-19 19:00:44', 2, '2024-11-19 19:00:00', 1)
    ]
    data_ICM_RSP_DATA_Columns = [
        (101, 0, 1),
        (102, 1, 0),
        (103, 0, 1),
        (104, 1, 1),
        (105, 1, 1),
        (106, 0, 0),
        (107, 1, 0)
    ]

    cursor.executemany("INSERT INTO ICM_EPI_Columns VALUES (?,?,?,?,?,?,?)", data_ICM_EPI_Columns)
    cursor.executemany("INSERT INTO ICM_RSP_Columns VALUES (?,?,?,?,?,?)", data_ICM_RSP_Columns)
    cursor.executemany("INSERT INTO ICM_RSP_DATA_Columns VALUES (?,?,?)", data_ICM_RSP_DATA_Columns)

    conn.commit()
    conn.close()