import re
import os
import subprocess

md_path = "/Users/nirajrajendranaphade/.gemini/antigravity/brain/6251d939-fc75-40ac-b6ef-2c32d7b6b8e5/project_visualizations.md"
out_dir = "/Users/nirajrajendranaphade/Programming/sih-drdo/docs/detailed"

with open(md_path, "r") as f:
    content = f.read()

# Find all mermaid blocks
matches = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
filenames = ["01_hardware_topology", "02_software_pipeline", "03_gtcrn_model"]

for i, mmd_code in enumerate(matches):
    base_name = filenames[i]
    mmd_path = os.path.join(out_dir, f"{base_name}.mmd")
    png_path = os.path.join(out_dir, f"{base_name}.png")
    
    # FIX: Quote the edge labels so Mermaid CLI parser doesn't choke on parenthesis
    mmd_code = re.sub(r'-->\|(.*?)\|', r'-->|"\1"|', mmd_code)
    mmd_code = re.sub(r'-\.->\|(.*?)\|', r'-.->|"\1"|', mmd_code)
    
    with open(mmd_path, "w") as f:
        f.write(mmd_code)
    
    print(f"Rendering {png_path}...")
    cmd = [
        "npx", "--yes", "@mermaid-js/mermaid-cli", 
        "-i", mmd_path, 
        "-o", png_path, 
        "-s", "4",
        "-b", "white"
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {png_path}")

print("All PNGs generated successfully.")
