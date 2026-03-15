import os

# Load API key
key = ""
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.strip().split("=")[1]
                break

# Read the notebook builder
with open("build_notebook.py", "r", encoding="utf-8") as f:
    code = f.read()

# Inject test configuration (5000 steps instead of 500,000, and inject the local API key)
code = code.replace("TRAIN_TIMESTEPS = 500_000", "TRAIN_TIMESTEPS = 5_000")
code = code.replace("OPENROUTER_API_KEY = \"\"", f"OPENROUTER_API_KEY = \"{key}\"")

# Change output path to avoid overwriting the real notebook
code = code.replace("drl_trading_pipeline.ipynb", "drl_trading_test.ipynb")

# Save and run
with open("build_notebook_test.py", "w", encoding="utf-8") as f:
    f.write(code)

os.system("python build_notebook_test.py")
os.system("jupyter nbconvert --to notebook --execute notebooks/drl_trading_test.ipynb --output drl_trading_test_executed.ipynb")
