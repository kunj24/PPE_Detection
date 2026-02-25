#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick database viewer - shows all data and exports to CSV files
"""

import sqlite3
import pandas as pd
import os

DB_FILE = "database.db"

def view_all_data():
    """View all data from both tables and export to CSV"""
    
    if not os.path.exists(DB_FILE):
        print(f"❌ Database file not found: {DB_FILE}")
        return
    
    conn = sqlite3.connect(DB_FILE)
    
    print("=" * 80)
    print("📋 WORKERS TABLE")
    print("=" * 80)
    
    workers_df = pd.read_sql_query("SELECT * FROM workers", conn)
    if workers_df.empty:
        print("No workers registered yet.\n")
    else:
        # Hide the face_encoding column (it's binary data)
        display_cols = [c for c in workers_df.columns if c != 'face_encoding']
        print(workers_df[display_cols].to_string(index=False))
        print(f"\n✓ Total workers: {len(workers_df)}")
        
        # Export to CSV
        workers_df[display_cols].to_csv("workers_export.csv", index=False)
        print(f"✓ Exported to: workers_export.csv\n")
    
    print("=" * 80)
    print("⚠️  VIOLATION LOGS TABLE")
    print("=" * 80)
    
    violations_df = pd.read_sql_query(
        "SELECT * FROM violation_logs ORDER BY timestamp DESC", conn
    )
    
    if violations_df.empty:
        print("No violations logged yet.\n")
    else:
        print(violations_df.to_string(index=False))
        print(f"\n✓ Total violations: {len(violations_df)}")
        
        # Export to CSV
        violations_df.to_csv("violations_export.csv", index=False)
        print(f"✓ Exported to: violations_export.csv\n")
    
    conn.close()
    
    print("=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    
    if not violations_df.empty:
        print(f"\n🔴 By Violation Type:")
        print(violations_df['violation_type'].value_counts().to_string())
        
        print(f"\n🚨 By Severity:")
        print(violations_df['severity_level'].value_counts().to_string())
        
        print(f"\n👤 By Worker:")
        print(violations_df['name'].value_counts().head(10).to_string())
        
        print(f"\n📅 By Date:")
        violations_df['date'] = pd.to_datetime(violations_df['timestamp']).dt.date
        print(violations_df['date'].value_counts().sort_index().to_string())
    
    print("\n" + "=" * 80)
    print("✅ Done! Check workers_export.csv and violations_export.csv")
    print("=" * 80)


if __name__ == "__main__":
    view_all_data()
