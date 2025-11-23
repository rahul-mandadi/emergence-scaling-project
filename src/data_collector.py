import yaml
import pandas as pd
import os
import random
import time
from tqdm import tqdm
import sys

# Import our custom modules from the 'src' folder
from api_clients import api_call_with_retry, call_gemini, call_llama
from prompt_builder import load_bbh_task, build_prompt
from response_parser import extract_answer

def estimate_tokens(prompt):
    """Rough token estimation for cost-benefit analysis."""
    return len(prompt.split()) * 1.3  # A rough estimate

def load_config(config_path='config.yaml'):
    """Loads the main YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment():
    """Main function to run the entire data collection process."""
    
    # 1. --- SETUP ---
    print("=" * 70)
    print(" DATA COLLECTOR - CoT Fix Re-Run".center(70))
    print("=" * 70)
    print("\nLoading configuration...")
    config = load_config()
    
    api_keys = config['api_keys']
    models_config = config['models']
    tasks = config['tasks']
    params = config['parameters']
    paths = config['paths']

    results = []
    
    # --- LOADING LOGIC ---
    # Priority 1: Load from checkpoint (resume mid-run)
    # Priority 2: Load from final_results.csv (resume after cleanup)
    checkpoint_path = os.path.join(paths['checkpoints_dir'], 'checkpoint_results.csv')
    final_results_path = paths['processed_output']
    
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found checkpoint: {checkpoint_path}")
        results_df = pd.read_csv(checkpoint_path)
        results = results_df.to_dict('records')
        print(f"✅ Loaded {len(results)} results from checkpoint")
        print(f"   (This checkpoint will be used to resume the run)")
    elif os.path.exists(final_results_path):
        print(f"\n📂 No checkpoint. Loading from: {final_results_path}")
        results_df = pd.read_csv(final_results_path)
        results = results_df.to_dict('records')
        print(f"✅ Loaded {len(results)} existing results")
        
        # Check what we have
        techniques_count = results_df['technique'].value_counts()
        print(f"\n📊 Current data breakdown:")
        for tech, count in techniques_count.items():
            print(f"   {tech}: {count} rows")
    else:
        print("\n🆕 No existing data found. Starting fresh.")
    
    # Build set of completed jobs (only those with valid responses)
    completed_jobs = set()
    for res in results:
        if pd.notnull(res.get('raw_response')) and res.get('raw_response') != "":
            completed_jobs.add((res['model_key'], res['technique'], res['task'], res['input']))
    
    total_jobs = len(models_config) * len(['zero-shot', 'few-shot', 'cot']) * len(tasks) * params['examples_per_task']
    remaining = total_jobs - len(completed_jobs)
    
    print(f"\n📊 Status:")
    print(f"   Total jobs: {total_jobs}")
    print(f"   Completed: {len(completed_jobs)}")
    print(f"   Remaining: {remaining}")
    
    if remaining == 0:
        print("\n✅ All jobs already complete! Nothing to do.")
        print(f"📊 Final results are in: {final_results_path}")
        return
    
    print(f"\n🚀 Starting experiment loop to collect {remaining} results...")
    print(f"   (This should be 150 CoT experiments with proper reasoning)\n")
    
    progress_bar = tqdm(total=total_jobs, initial=len(completed_jobs))
    
    # 2. --- MAIN EXPERIMENT LOOP ---
    try:
        for model_key, model_name in models_config.items():
            for technique in ['zero-shot', 'few-shot', 'cot']:
                for task in tasks:
                    
                    task_data = load_bbh_task(task, paths['bbh_dataset_dir'])
                    if not task_data:
                        print(f"⚠️  Skipping task {task} due to load error.")
                        continue
                    
                    random.seed(params['seed'])
                    shuffled_examples = random.sample(task_data['examples'], len(task_data['examples']))
                    
                    test_examples = shuffled_examples[:params['examples_per_task']]
                    demo_examples = shuffled_examples[params['examples_per_task'] : 
                                                      params['examples_per_task'] + params['few_shot_examples']]
                    
                    for example in test_examples:
                        example_input = example['input']
                        example_target = example['target']
                        
                        # Skip if already completed
                        if (model_key, technique, task, example_input) in completed_jobs:
                            continue
                        
                        # BUILD PROMPT WITH TASK NAME FOR COT
                        prompt = build_prompt(
                            question=example_input, 
                            technique=technique, 
                            demo_examples=demo_examples,
                            task_name=task if technique == 'cot' else None  # Pass task_name for CoT
                        )
                        
                        input_tokens = estimate_tokens(prompt)
                        
                        api_args = { 'prompt': prompt, 'temperature': params['temperature'] }
                        
                        # Update progress bar
                        progress_bar.update(1)
                        
                        # Sleep before API call (increased for stability)
                        if model_key == 'gemini-pro':
                            time.sleep(15)  # Google: 4 req/min to be safe
                        else:
                            time.sleep(2)   # Groq: give it breathing room
                        
                        # Set up API call
                        if model_key == 'gemini-pro':
                            api_func = call_gemini
                            api_args['api_key'] = api_keys['google']
                        else:
                            api_func = call_llama
                            api_args['model_name'] = model_name
                            api_args['api_key'] = api_keys['groq']
                        
                        try:
                            raw_response = api_call_with_retry(api_func, **api_args)
                            job_key = (model_key, technique, task, example_input)
                            completed_jobs.add(job_key)
                            
                            # Debug: Show first CoT response to verify it's working
                            if technique == 'cot' and len([r for r in results if r.get('technique') == 'cot']) == 0:
                                print(f"\n🔍 First CoT response preview (first 300 chars):")
                                print(f"   Model: {model_key}, Task: {task}")
                                print(f"   Response: {raw_response[:300]}...")
                                print(f"   (Checking that model is actually reasoning step-by-step)\n")

                        except Exception as e:
                            print(f"\n❌ API call failed: {model_key}/{technique}/{task}")
                            print(f"   Error: {str(e)[:100]}")
                            raw_response = ""  # Log failure
                        
                        result_data = {
                            'model_key': model_key,
                            'model_name': model_name,
                            'technique': technique,
                            'task': task,
                            'input': example_input,
                            'target': example_target,
                            'prediction': extract_answer(raw_response, task),
                            'correct': (extract_answer(raw_response, task) == example_target),
                            'raw_response': raw_response,
                            'input_tokens': input_tokens
                        }
                        results.append(result_data)
                        
                        # Save checkpoint every 10 results
                        if len(results) % params['checkpoint_interval'] == 0:
                            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                            # Quick stats update
                            df_temp = pd.DataFrame(results)
                            cot_count = len(df_temp[df_temp['technique'] == 'cot'])
                            progress_bar.set_description(f"CoT collected: {cot_count}/150")
                        
                        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    else:
        # SUCCESS: Try block completed without errors
        progress_bar.close()
        print("\n" + "=" * 70)
        print(" EXPERIMENT COMPLETE".center(70))
        print("=" * 70)
        
        # Final save
        pd.DataFrame(results).to_csv(paths['processed_output'], index=False)
        print(f"\n✅ Final results saved to: {paths['processed_output']}")
        print(f"📊 Total results: {len(results)}")
        
        # Show final breakdown
        df_final = pd.DataFrame(results)
        print(f"\n📊 Final technique breakdown:")
        for tech, count in df_final['technique'].value_counts().items():
            print(f"   {tech}: {count} rows")
        
        # Verify CoT is now using proper reasoning
        cot_sample = df_final[df_final['technique'] == 'cot'].iloc[0] if len(df_final[df_final['technique'] == 'cot']) > 0 else None
        if cot_sample is not None:
            print(f"\n🔍 Sample CoT response check (first 200 chars):")
            print(f"   {cot_sample['raw_response'][:200]}...")
            if "step" in cot_sample['raw_response'].lower()[:200]:
                print("   ✅ Confirmed: Models are now doing step-by-step reasoning!")
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("\n🗑️  Checkpoint file removed.")
    
    finally:
        # ALWAYS runs (error, interrupt, or success)
        if 'else' not in locals():  # If success block didn't run
            progress_bar.close()
            print("\n" + "=" * 70)
            print(" SAVING PROGRESS".center(70))
            print("=" * 70)
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            print(f"📊 Results collected: {len(results)}")
            print(f"🔄 Re-run the script to resume from this checkpoint.")
        
        print("\n🏁 Script shutdown complete.\n")

# --- Main entry point ---
if __name__ == "__main__":
    run_experiment()