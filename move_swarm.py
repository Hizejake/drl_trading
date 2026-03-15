with open('build_notebook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_swarm = -1
end_swarm = -1
for i, line in enumerate(lines):
    if '# CELL 15: LLM Swarm (optional)' in line:
        start_swarm = i - 1  # The border
    if 'print("No OPENROUTER_API_KEY set. The RL environment will default to using randomly generated macro vectors.")' in line:
        end_swarm = i + 2  # The closing triple quote and newline

if start_swarm != -1 and end_swarm != -1:
    swarm_block = lines[start_swarm:end_swarm]
    del lines[start_swarm:end_swarm]
    
    insert_idx = -1
    for i, line in enumerate(lines):
        if '# CELL 6: CVML Module' in line:
            insert_idx = i - 1  # The top border of Cell 6
            break
            
    if insert_idx != -1:
        lines = lines[:insert_idx] + swarm_block + ["\n"] + lines[insert_idx:]
        with open('build_notebook.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Successfully moved swarm block!')
    else:
        print('Could not find insert index')
else:
    print('Could not find swarm block bounds', start_swarm, end_swarm)
