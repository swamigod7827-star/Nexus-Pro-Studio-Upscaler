import json
import os

for file in ['Cloud_Workspaces/Kaggle_Setup.ipynb', 'Cloud_Workspaces/Colab_Setup.ipynb']:
    with open(file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                # Update git clone to remove existing repo first
                if '!git clone $GITHUB_REPO_URL' in line:
                    line = line.replace('!git clone $GITHUB_REPO_URL', '!rm -rf $repo_name && git clone $GITHUB_REPO_URL')
                # Update app.py to cloud_worker.py
                if '"app.py"' in line:
                    line = line.replace('"app.py"', '"cloud_worker.py"')
                if "'app.py'" in line:
                    line = line.replace("'app.py'", "'cloud_worker.py'")
                new_source.append(line)
            cell['source'] = new_source
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
print("Notebooks updated successfully!")
