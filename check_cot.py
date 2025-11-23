import pandas as pd

df = pd.read_csv('data/processed/final_results.csv')

# Check a few CoT responses for proper reasoning
cot_df = df[df['technique'] == 'cot']
print('Checking CoT responses for step-by-step reasoning:')
print('='*60)

for i in range(min(5, len(cot_df))):
    sample = cot_df.iloc[i]
    print(f'\nModel: {sample["model_key"]}, Task: {sample["task"]}')
    print(f'Response length: {len(sample["raw_response"])}')
    step_count = sample['raw_response'].lower().count('step')
    print(f'Contains "step": {step_count} times')
    print(f'First 300 chars: {sample["raw_response"][:300]}')
    print('-'*40)

# Also check if CoT responses are longer than zero-shot
print('\n' + '='*60)
print('Response length comparison:')
avg_lengths = df.groupby('technique')['raw_response'].apply(lambda x: x.str.len().mean())
for tech, length in avg_lengths.items():
    print(f'{tech}: {length:.0f} chars avg')

# Check parsing success rate by technique
print('\n' + '='*60)
print('Parsing success rate by technique:')
parse_success = df.groupby('technique')['prediction'].apply(lambda x: (x != '').mean() * 100)
for tech, rate in parse_success.items():
    print(f'{tech}: {rate:.1f}%')

# Check if CoT hurts specific tasks more
print('\n' + '='*60)
print('CoT vs Zero-shot by task:')
for task in df['task'].unique():
    cot_acc = df[(df['task']==task) & (df['technique']=='cot')]['correct'].mean()
    zs_acc = df[(df['task']==task) & (df['technique']=='zero-shot')]['correct'].mean()
    diff = cot_acc - zs_acc
    print(f'{task[:30]:30} CoT: {cot_acc:.1%} vs ZS: {zs_acc:.1%} (diff: {diff:+.1%})')
