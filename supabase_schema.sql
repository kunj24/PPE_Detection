-- ============================================================
-- PPE Detection System – Supabase Schema
-- Run this entire file inside: Supabase → SQL Editor → New query
-- ============================================================

-- 1. Workers table
CREATE TABLE IF NOT EXISTS workers (
    id            BIGSERIAL     PRIMARY KEY,
    employee_id   TEXT          UNIQUE NOT NULL,
    name          TEXT          NOT NULL,
    department    TEXT          NOT NULL,
    image_path    TEXT,
    face_encoding TEXT,                          -- base64-encoded pickle bytes
    created_at    TIMESTAMPTZ   DEFAULT NOW()
);

-- 2. Violation logs table
CREATE TABLE IF NOT EXISTS violation_logs (
    id                  BIGSERIAL     PRIMARY KEY,
    timestamp           TIMESTAMPTZ   DEFAULT NOW(),
    employee_id         TEXT,
    name                TEXT,
    department          TEXT,
    violation_type      TEXT          NOT NULL,
    confidence          NUMERIC(5,2)  NOT NULL,
    image_snapshot_path TEXT,                    -- Supabase Storage public URL
    camera_location     TEXT          DEFAULT 'Main Camera',
    severity_level      TEXT          DEFAULT 'Low',
    status              TEXT          DEFAULT 'Open'
);

-- 3. Indexes for fast dashboard queries
CREATE INDEX IF NOT EXISTS idx_violations_timestamp    ON violation_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_violations_employee     ON violation_logs (employee_id);
CREATE INDEX IF NOT EXISTS idx_violations_type         ON violation_logs (violation_type);

-- 4. Enable Row Level Security (keeps data private – service key bypasses this)
ALTER TABLE workers        ENABLE ROW LEVEL SECURITY;
ALTER TABLE violation_logs ENABLE ROW LEVEL SECURITY;

-- Allow full access via the service/anon key used by the app
CREATE POLICY "allow_all_workers"        ON workers        FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_violation_logs" ON violation_logs FOR ALL USING (true) WITH CHECK (true);
