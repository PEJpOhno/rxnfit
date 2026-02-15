import json

with open('example_script/example_script_classRxnODEsolver_1b.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    cell_id = c.get('id', 'N/A')[:30] if 'id' in c else 'N/A'
    source_len = len(c.get('source', []))
    is_empty = source_len == 0
    print(f'Cell {i}: type={c["cell_type"]}, id={cell_id}, source_len={source_len}, empty={is_empty}')
