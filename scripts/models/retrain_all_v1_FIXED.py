import subprocess
import os

print("🚀 RETRAINING ALL MODELS ON V1_IMPUTED_FIXED (LEAKAGE-FREE)\n")
print("="*70)

# 📁 Visualization output folder
VISU_DIR = "/Users/lyna.thc/Desktop/thesis/visu"
os.makedirs(VISU_DIR, exist_ok=True)

# 📁 Results folder (optional but clean)
RESULTS_DIR = "/Users/lyna.thc/Desktop/thesis/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the models to retrain
models = [
    ("Random Forest (tuned)", "RANDOM_FOREST_TUNED.py"),
    ("XGBoost (tuned)", "XGBOOST_TUNED.py"),
    ("XGBoost (untuned)", "XGBOOST.py")
]

# NEW CLEAN dataset path
new_dataset = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_v1_IMPUTED_FIXED.csv"

print(f"📦 NEW CLEAN dataset: {new_dataset}")
print(f"📊 Size: 6,001 flights (leakage-free)\n")

for i, (model_name, script_name) in enumerate(models, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/3] Training {model_name}...")
    print(f"{'='*70}\n")
    
    script_path = f"scripts/models/{script_name}"
    
    # Read the script
    with open(script_path, 'r') as f:
        code = f.read()
    
    # 🔁 Replace dataset paths
    replacements = [
        ('DATA_PATH = "/Users/lyna.thc/Desktop/thesis/data/processed/flights_with_weather_road_history.csv"', 
         f'DATA_PATH = "{new_dataset}"'),
        ("DATA_PATH = 'data/processed/flights_with_weather_road_history.csv'", 
         f"DATA_PATH = '{new_dataset}'"),
        ('flights_with_weather_road_history.csv', 
         'flights_v1_IMPUTED_FIXED.csv'),
        ('flights_v1_IMPUTED.csv', 
         'flights_v1_IMPUTED_FIXED.csv')
    ]
    
    for old, new in replacements:
        code = code.replace(old, new)

    # 🧠 Inject visualization + saving logic
    injection = f"""
import os
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

VISU_DIR = "{VISU_DIR}"
RESULTS_DIR = "{RESULTS_DIR}"

def save_plot(name):
    path = os.path.join(VISU_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    print(f"📊 Saved visualization: {{path}}")
    plt.close()
"""

    code = injection + "\n" + code

    # 🧠 Auto-replace plt.show() → save_plot
    code = code.replace("plt.show()", "save_plot(f'{script_name}_plot.png')")

    # Create temporary script
    temp_script = f"/tmp/temp_{script_name}"
    with open(temp_script, 'w') as f:
        f.write(code)
    
    # Run it
    try:
        result = subprocess.run(
            ['python', temp_script],
            capture_output=True,
            text=True,
            cwd='/Users/lyna.thc/Desktop/thesis'
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
        
        print(f"✅ {model_name} complete!\n")
        
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}\n")
    
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

print("\n" + "="*70)
print("🎉 ALL MODELS RETRAINED + VISUALIZATIONS SAVED!")
print("="*70)
print(f"\n📊 Visualizations → {VISU_DIR}")