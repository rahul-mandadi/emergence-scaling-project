import pandas as pd
import os

print("=" * 70)
print(" REMOVING CoT ROWS FOR RE-RUN".center(70))
print("=" * 70)

# Load current data
data_path = 'data/processed/final_results.csv'

if not os.path.exists(data_path):
    print(f"\n❌ ERROR: {data_path} not found!")
    print("Make sure you're in the project root directory.")
    exit(1)

print(f"\n📂 Loading data from: {data_path}")
df = pd.read_csv(data_path)

print(f"\n📊 CURRENT STATE:")
print(f"   Total rows: {len(df)}")
print(f"\n   Breakdown by technique:")
print(df['technique'].value_counts().sort_index())

# Count CoT rows to be removed
cot_count = len(df[df['technique'] == 'cot'])
print(f"\n🗑️  Rows to remove: {cot_count} (CoT experiments)")

# Keep everything EXCEPT CoT
df_cleaned = df[df['technique'] != 'cot'].copy()

print(f"\n📊 AFTER REMOVAL:")
print(f"   Remaining rows: {len(df_cleaned)}")
print(f"   Expected after re-run: {len(df_cleaned) + 150}")

print(f"\n   Breakdown of remaining data:")
print(df_cleaned['technique'].value_counts().sort_index())

# Verify we still have all models and tasks
print(f"\n✅ VERIFICATION:")
print(f"   Models remaining: {df_cleaned['model_key'].nunique()} (expected: 3)")
print(f"   Tasks remaining: {df_cleaned['task'].nunique()} (expected: 5)")
print(f"   Zero-shot rows: {len(df_cleaned[df_cleaned['technique']=='zero-shot'])} (expected: 150)")
print(f"   Few-shot rows: {len(df_cleaned[df_cleaned['technique']=='few-shot'])} (expected: 150)")

# Save the cleaned data
df_cleaned.to_csv(data_path, index=False)

print(f"\n💾 SAVED: Cleaned data written to {data_path}")
print(f"\n✅ Ready to re-run data_collector.py!")
print(f"   It will collect {cot_count} new CoT experiments with the corrected prompts.")
print("\n" + "=" * 70)