# Reproduce MetaGPT DI Benchmark Workflow

## 0) Clone Project & Download Dataset

```bash
git clone https://github.com/FoundationAgents/MetaGPT.git
cd MetaGPT
```

Dataset download link:

- https://drive.google.com/drive/folders/17SpI9WL9kzd260q2DArbXKNcqhidjA7s?usp=sharing

After downloading, place the dataset under:

```text
MetaGPT/data/di_dataset/
```

Expected structure (minimum):

```text
data/di_dataset/ml_benchmark/04_titanic/
```

---

## 1) Notebook A: `ml_benchmark_runner.ipynb` (Run + Persist + Visualize)

> This section integrates the code cells in execution order.

### 1.1 Setup output directory

```python
# Setup output directories for results
from pathlib import Path
from datetime import datetime

OUTPUT_BASE_DIR = Path('/Users/tian/CUHK/FTEC-5660/project/MetaGPT/benchmark_results')
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = OUTPUT_BASE_DIR / f"run_{TIMESTAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = RUN_DIR / 'results.json'
RESULTS_CSV = RUN_DIR / 'results.csv'
REPORT_MD = RUN_DIR / 'report.md'

print(f"✓ Output directory: {RUN_DIR}")
print(f"  - Results JSON: {RESULTS_JSON.name}")
print(f"  - Results CSV: {RESULTS_CSV.name}")
print(f"  - Report MD: {REPORT_MD.name}")
```

### 1.2 Imports

```python
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Data and ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MetaGPT imports
sys.path.insert(0, '/Users/tian/CUHK/FTEC-5660/project/MetaGPT')
from examples.di.requirements_prompt import ML_BENCHMARK_REQUIREMENTS
from metagpt.const import DATA_PATH
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.tools.tool_recommend import TypeMatchToolRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print(f"✓ All imports successful")
print(f"✓ DATA_PATH: {DATA_PATH}")
print(f"✓ Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

### 1.3 Validate benchmark dataset structure

```python
# Define benchmark tasks and their expected metrics
BENCHMARK_TASKS = {
    "04_titanic": {"metric": "accuracy", "description": "Titanic Passenger Survival Prediction"},
    "05_house_prices": {"metric": "rmse", "description": "House Prices Prediction (Log RMSE)"},
    "06_santander_customer": {"metric": "auc", "description": "Santander Customer Transaction Prediction"},
    "07_icr_identify": {"metric": "f1_score", "description": "ICR Age-Related Conditions Prediction"},
    "08_santander_value": {"metric": "rmsle", "description": "Santander Value Prediction (RMSLE)"},
}

# Validate dataset structure
di_dataset_path = Path(DATA_PATH) / "di_dataset" / "ml_benchmark"
print(f"Checking ML Benchmark dataset structure...")
print(f"Dataset path: {di_dataset_path}")

if di_dataset_path.exists():
    tasks_found = []
    for task_dir in di_dataset_path.iterdir():
        if task_dir.is_dir():
            tasks_found.append(task_dir.name)
    print(f"✓ Found {len(tasks_found)} task directories:")
    for task in sorted(tasks_found):
        print(f"  - {task}")
else:
    print(f"✗ Dataset not found at {di_dataset_path}")
    print(f"  Please ensure you have downloaded di_dataset to {DATA_PATH}")
```

### 1.4 Configure benchmark runner

```python
class BenchmarkRunner:
    """Utility class to run benchmarks and persist results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results = []
        self.start_time = None
        self.end_time = None
        
    def create_data_interpreter(self, use_reflection: bool = True) -> DataInterpreter:
        """Create a configured DataInterpreter instance"""
        return DataInterpreter(
            use_reflection=use_reflection,
            tool_recommender=TypeMatchToolRecommender(tools=["<all>"])
        )
    
    def save_results_json(self):
        """Save results to JSON file"""
        with open(RESULTS_JSON, 'w') as f:
            json.dump({
                'timestamp': self.start_time.isoformat() if self.start_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if (self.start_time and self.end_time) else None,
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to {RESULTS_JSON}")
    
    def save_results_csv(self):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"✓ Results saved to {RESULTS_CSV}")
    
    def add_result(self, task_name: str, status: str, metrics: Dict = None, error: str = None, duration: float = None):
        """Add a result record"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'task_name': task_name,
            'status': status,
            'duration_seconds': duration,
            'error': error,
        }
        if metrics:
            record.update(metrics)
        self.results.append(record)

# Initialize benchmark runner
runner = BenchmarkRunner(RUN_DIR)
print(f"✓ BenchmarkRunner initialized")
print(f"  Available tasks: {', '.join(BENCHMARK_TASKS.keys())}")
```

### 1.5 Configure tasks

```python
# Configuration for benchmark runs
CONFIG = {
    "use_reflection": False,  # set True/False for ablation
    "data_dir": DATA_PATH,
    "tasks_to_run": ["04_titanic"],
}

# Validate selected tasks
print(f"Benchmark Configuration:")
print(f"  Use Reflection: {CONFIG['use_reflection']}")
print(f"  Data Dir: {CONFIG['data_dir']}")
print(f"  Tasks to run: {CONFIG['tasks_to_run']}")
print()

# Validate tasks exist in dataset
for task in CONFIG['tasks_to_run']:
    task_path = Path(CONFIG['data_dir']) / "di_dataset" / "ml_benchmark" / task
    exists = task_path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {task}: {task_path}")
```

### 1.6 Run benchmark tasks

```python
async def run_benchmark_task(task_name: str, config: Dict) -> Dict:
    """
    Run a single benchmark task and capture results
    """
    print(f"\n{'='*60}")
    print(f"Running task: {task_name}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    task_start = asyncio.get_event_loop().time()
    
    try:
        if task_name not in ML_BENCHMARK_REQUIREMENTS:
            raise ValueError(f"Task {task_name} not found in ML_BENCHMARK_REQUIREMENTS")
        
        requirement = ML_BENCHMARK_REQUIREMENTS[task_name].format(data_dir=config['data_dir'])
        di = runner.create_data_interpreter(use_reflection=config['use_reflection'])
        
        print(f"Executing Data Interpreter for {task_name}...")
        await di.run(requirement)
        
        task_end = asyncio.get_event_loop().time()
        duration = task_end - task_start
        
        runner.add_result(
            task_name=task_name,
            status='success',
            duration=duration,
            metrics={
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
            }
        )
        
        print(f"✓ Task completed in {duration:.2f} seconds")
        return {'status': 'success', 'duration': duration}
        
    except Exception as e:
        task_end = asyncio.get_event_loop().time()
        duration = task_end - task_start
        error_msg = str(e)
        print(f"✗ Task failed: {error_msg}")
        
        runner.add_result(
            task_name=task_name,
            status='failed',
            duration=duration,
            error=error_msg,
        )
        
        return {'status': 'failed', 'duration': duration, 'error': error_msg}

async def run_all_benchmarks():
    """Run all configured benchmark tasks"""
    runner.start_time = datetime.now()
    
    print(f"\n{'#'*60}")
    print(f"Starting ML Benchmark Run")
    print(f"Timestamp: {runner.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    for task_name in CONFIG['tasks_to_run']:
        result = await run_benchmark_task(task_name, CONFIG)
    
    runner.end_time = datetime.now()
    
    print(f"\n{'#'*60}")
    print(f"Benchmark Run Complete")
    print(f"Total Duration: {(runner.end_time - runner.start_time).total_seconds():.2f} seconds")
    print(f"{'#'*60}\n")

print(f"Starting benchmark execution...")
await run_all_benchmarks()
```

### 1.7 Persist results

```python
# Save results to persistent storage
print(f"\nSaving results to files...")
runner.save_results_json()
runner.save_results_csv()

# Display results summary
if runner.results:
    df_results = pd.DataFrame(runner.results)
    print(f"\n{'='*80}")
    print(f"Results Summary ({len(runner.results)} tasks)")
    print(f"{'='*80}")
    print(df_results.to_string(index=False))
else:
    print("No results to display yet. Please run benchmarks first.")
```

### 1.8 Visualize results

```python
def create_visualizations(results: List[Dict], output_dir: Path):
    """Create visualizations for benchmark results"""
    if not results:
        print("No results to visualize")
        return
    
    df = pd.DataFrame(results)
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    status_counts = df['status'].value_counts()
    colors = ['#2ecc71' if x == 'success' else '#e74c3c' for x in status_counts.index]
    status_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Task Status Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Status')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    ax2 = plt.subplot(2, 2, 2)
    df_successful = df[df['status'] == 'success'].copy()
    if not df_successful.empty:
        df_successful['duration_or_0'] = df_successful['duration_seconds'].fillna(0)
        df_successful.plot(x='task_name', y='duration_or_0', kind='bar', ax=ax2, legend=False, color='#3498db')
        ax2.set_title('Execution Time by Task', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_xlabel('Task')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    ax3 = plt.subplot(2, 2, 3)
    if 'timestamp' in df.columns:
        df['task_num'] = range(len(df))
        ax3.scatter(df['task_num'], df['duration_seconds'].fillna(0), s=100, alpha=0.6, color='#9b59b6')
        ax3.set_title('Task Duration Timeline', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Duration (seconds)')
        ax3.set_xlabel('Task Execution Order')
        ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    total_tasks = len(df)
    successful_tasks = len(df[df['status'] == 'success'])
    failed_tasks = len(df[df['status'] == 'failed'])
    total_duration = df['duration_seconds'].sum() if 'duration_seconds' in df.columns else 0
    
    summary_text = f"""
    BENCHMARK SUMMARY
    {'─' * 40}
    Total Tasks: {total_tasks}
    Successful: {successful_tasks} ✓
    Failed: {failed_tasks} ✗
    Success Rate: {(successful_tasks/total_tasks*100):.1f}%
    Total Duration: {total_duration:.2f}s
    Avg Duration: {(total_duration/max(1, successful_tasks)):.2f}s
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    viz_path = output_dir / 'benchmark_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {viz_path}")
    plt.show()

if runner.results:
    create_visualizations(runner.results, RUN_DIR)
else:
    print("No results available for visualization yet.")
```

### 1.9 Generate markdown report

```python
def generate_markdown_report(results: List[Dict], output_dir: Path) -> str:
    """Generate a comprehensive Markdown report"""
    
    df = pd.DataFrame(results) if results else pd.DataFrame()
    
    report = f"""# ML Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report documents the execution of MetaGPT Data Interpreter ML benchmark tasks.

### Key Metrics

"""
    
    if not df.empty:
        total_tasks = len(df)
        successful = len(df[df['status'] == 'success'])
        failed = len(df[df['status'] == 'failed'])
        success_rate = (successful / total_tasks * 100) if total_tasks > 0 else 0
        total_time = df['duration_seconds'].sum() if 'duration_seconds' in df.columns else 0
        
        report += f"""
- **Total Tasks:** {total_tasks}
- **Successful:** {successful} ✓
- **Failed:** {failed} ✗
- **Success Rate:** {success_rate:.1f}%
- **Total Duration:** {total_time:.2f} seconds
- **Average Duration per Task:** {(total_time/max(1, successful)):.2f} seconds

## Task Results

"""
        
        report += "| Task Name | Status | Duration (s) | Error |\n"
        report += "|-----------|--------|--------------|-------|\n"
        
        for _, row in df.iterrows():
            task_name = row.get('task_name', 'N/A')
            status = row.get('status', 'N/A')
            duration = f"{row.get('duration_seconds', 0):.2f}" if pd.notna(row.get('duration_seconds')) else "N/A"
            error = row.get('error', '').replace('\n', ' ')[:50] if row.get('error') else "None"
            status_icon = "✓" if status == 'success' else "✗"
            report += f"| {task_name} | {status_icon} {status} | {duration} | {error} |\n"
        
        report += f"""

## Configuration

- **Use Reflection:** {CONFIG.get('use_reflection', 'N/A')}
- **Data Directory:** {CONFIG.get('data_dir', 'N/A')}
- **Tasks Run:** {', '.join(CONFIG.get('tasks_to_run', []))}

## Output Files

- **Results JSON:** `results.json` - Machine-readable results
- **Results CSV:** `results.csv` - Tabular format results
- **Visualization:** `benchmark_visualization.png` - Performance charts
- **Report:** `report.md` - This report

## Recommendations

"""
        
        if failed > 0:
            report += f"- **Action Required:** {failed} task(s) failed. Review error messages above for details.\n"
        
        if success_rate >= 90:
            report += "- ✓ Excellent success rate. All systems operating normally.\n"
        elif success_rate >= 50:
            report += "- ⚠ Moderate success rate. Investigate failing tasks for improvements.\n"
        else:
            report += "- ✗ Low success rate. Review configuration and dataset availability.\n"
        
        report += f"""
## System Information

- **Execution Time:** {runner.start_time.strftime('%Y-%m-%d %H:%M:%S') if runner.start_time else 'N/A'}
- **MetaGPT Data Path:** {DATA_PATH}
- **Output Directory:** {output_dir}

---
*Report generated automatically by ML Benchmark Runner*
"""
    
    return report

if runner.results:
    report_content = generate_markdown_report(runner.results, RUN_DIR)
    with open(REPORT_MD, 'w') as f:
        f.write(report_content)
    print(f"✓ Report saved to {REPORT_MD}\n")
    print(report_content)
else:
    print("No results to generate report from.")
```

---

## 2) Notebook B: `ablation_reflection_analysis.ipynb` (Post-process True vs False)

### 2.1 Imports and directories

```python
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path('/Users/tian/CUHK/FTEC-5660/project/MetaGPT')
BENCH_DIR = BASE_DIR / 'benchmark_results'
OUT_DIR = BENCH_DIR / 'ablation_summary'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Benchmark dir: {BENCH_DIR}')
print(f'Output dir: {OUT_DIR}')
```

### 2.2 Load all runs

```python
def parse_use_reflection(report_path: Path):
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r'\*\*Use Reflection:\*\*\s*(True|False)', text)
    if not m:
        return None
    return True if m.group(1) == 'True' else False

rows = []
for run_dir in sorted(BENCH_DIR.glob('run_*')):
    results_json = run_dir / 'results.json'
    report_md = run_dir / 'report.md'
    if not results_json.exists():
        continue

    use_reflection = parse_use_reflection(report_md)
    payload = json.loads(results_json.read_text(encoding='utf-8'))

    for item in payload.get('results', []):
        rows.append({
            'run_dir': run_dir.name,
            'use_reflection': use_reflection,
            'task_name': item.get('task_name'),
            'status': item.get('status'),
            'duration_seconds': item.get('duration_seconds'),
            'timestamp': item.get('timestamp'),
            'error': item.get('error'),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise ValueError('No benchmark results found under benchmark_results/run_*/results.json')

print('Loaded rows:', len(df))
print(df[['run_dir', 'use_reflection', 'task_name', 'status', 'duration_seconds']].to_string(index=False))
```

### 2.3 Export master table

```python
master_path = OUT_DIR / 'ablation_master_table.csv'
df.to_csv(master_path, index=False)
print(f'Saved: {master_path}')

df.head()
```

### 2.4 Group statistics

```python
summary = (
    df.groupby('use_reflection', dropna=False)
      .agg(
          n_runs=('status', 'count'),
          n_success=('status', lambda s: (s == 'success').sum()),
          success_rate=('status', lambda s: (s == 'success').mean()),
          duration_mean=('duration_seconds', 'mean'),
          duration_std=('duration_seconds', 'std'),
          duration_min=('duration_seconds', 'min'),
          duration_max=('duration_seconds', 'max'),
      )
      .reset_index()
      .sort_values('use_reflection', ascending=False)
)

summary_path = OUT_DIR / 'ablation_summary_table.csv'
summary.to_csv(summary_path, index=False)

print(f'Saved: {summary_path}')
summary
```

### 2.5 Ablation visualization

```python
plot_df = summary.copy()
plot_df['group'] = plot_df['use_reflection'].map({True: 'Reflection=True', False: 'Reflection=False'}).fillna('Unknown')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(plot_df['group'], plot_df['success_rate'], color=['#4e79a7', '#f28e2b'][:len(plot_df)])
axes[0].set_title('Success Rate by Group')
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('Success Rate')
for i, val in enumerate(plot_df['success_rate']):
    axes[0].text(i, val + 0.02, f'{val:.2f}', ha='center')

x = np.arange(len(plot_df))
axes[1].bar(x, plot_df['duration_mean'], yerr=plot_df['duration_std'].fillna(0), capsize=6, color=['#4e79a7', '#f28e2b'][:len(plot_df)])
axes[1].set_xticks(x)
axes[1].set_xticklabels(plot_df['group'])
axes[1].set_title('Duration Mean ± Std')
axes[1].set_ylabel('Seconds')
for i, val in enumerate(plot_df['duration_mean']):
    axes[1].text(i, val + 0.8, f'{val:.1f}s', ha='center')

plt.tight_layout()
fig_path = OUT_DIR / 'ablation_comparison.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f'Saved: {fig_path}')
plt.show()
```

### 2.6 Compute effect size

```python
true_mean = summary.loc[summary['use_reflection'] == True, 'duration_mean']
false_mean = summary.loc[summary['use_reflection'] == False, 'duration_mean']

if len(true_mean) and len(false_mean):
    delta = float(true_mean.iloc[0] - false_mean.iloc[0])
    faster = 'Reflection=False' if delta > 0 else 'Reflection=True'
    print(f'Mean duration delta (True - False): {delta:.2f} seconds')
    print(f'Faster group by mean duration: {faster}')
else:
    print('Cannot compute delta: missing one group.')
```

### 2.7 Export report snippet

```python
row_true = summary[summary['use_reflection'] == True].iloc[0] if (summary['use_reflection'] == True).any() else None
row_false = summary[summary['use_reflection'] == False].iloc[0] if (summary['use_reflection'] == False).any() else None

def fmt_row(name, row):
    if row is None:
        return f'| {name} | - | - | - | - |'
    return f"| {name} | {int(row['n_runs'])} | {row['success_rate']:.2%} | {row['duration_mean']:.2f} | {0.0 if pd.isna(row['duration_std']) else row['duration_std']:.2f} |"

md = []
md.append('## Ablation: `use_reflection` (04_titanic)')
md.append('')
md.append('| Group | Runs | Success Rate | Mean Duration (s) | Std Duration (s) |')
md.append('|---|---:|---:|---:|---:|')
md.append(fmt_row('Reflection=True', row_true))
md.append(fmt_row('Reflection=False', row_false))

if row_true is not None and row_false is not None:
    delta = row_true['duration_mean'] - row_false['duration_mean']
    md.append('')
    md.append(f"- Duration delta (True - False): **{delta:.2f}s**")
    md.append('- Note: Official benchmark number is not explicitly reported for this subset; this is a controlled internal comparison.')

md_text = '\n'.join(md)
print(md_text)

md_path = OUT_DIR / 'ablation_report_snippet.md'
md_path.write_text(md_text, encoding='utf-8')
print(f'\nSaved: {md_path}')
```

---

## 3) Minimal Reproduction Commands (Alternative)

If you only need to reproduce the exact benchmark workflow quickly:

```bash
# Run single benchmark with reflection on/off by editing CONFIG in notebook
# Then execute all cells in:
# 1) ml_benchmark_runner.ipynb
# 2) ablation_reflection_analysis.ipynb
```

Outputs will be saved under:

- `benchmark_results/run_YYYYMMDD_HHMMSS/`
- `benchmark_results/ablation_summary/`

