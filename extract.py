import re

with open('ai_upscaler.html', 'r', encoding='utf-8') as f:
    text = f.read()

ids = re.findall(r'id=[\'\"]([^\'\"]+)[\'\"]', text)

with open('ids.txt', 'w', encoding='utf-8') as f:
    for id in sorted(set(ids)):
        f.write(id + '\n')
