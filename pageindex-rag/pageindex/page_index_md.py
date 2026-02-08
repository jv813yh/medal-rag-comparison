import asyncio
import os
import re
from .utils import count_tokens, generate_summaries_for_structure, write_node_id

async def md_to_tree(md_path, if_thinning='no', min_token_threshold=5000, if_add_node_summary='yes', summary_token_threshold=200, model="gpt-4o", if_add_doc_description='no', if_add_node_text='yes'):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple markdown header parsing
    lines = content.split('\n')
    nodes = []
    current_node = None
    
    for line in lines:
        match = re.match(r'^(#+)\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2)
            node = {'title': title, 'level': level, 'text': '', 'nodes': []}
            nodes.append(node)
            current_node = node
        elif current_node:
            current_node['text'] += line + '\n'

    # Build hierarchy
    root_nodes = []
    stack = []
    for node in nodes:
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()
        if stack:
            stack[-1]['nodes'].append(node)
        else:
            root_nodes.append(node)
        stack.append(node)

    write_node_id(root_nodes)
    
    if if_add_node_summary == 'yes':
        await generate_summaries_for_structure(root_nodes, model=model)

    return {
        'doc_name': os.path.basename(md_path),
        'structure': root_nodes
    }
