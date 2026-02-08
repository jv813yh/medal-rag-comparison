from .page_index import *
from .page_index_md import md_to_tree
import json
import os
import asyncio

class PageIndex:
    def __init__(self):
        self.tree = {"docs": []}

    def index(self, input_path, model="gpt-4o"):
        print(f"Indexing documents in {input_path}...")
        if os.path.isdir(input_path):
            files = [f for f in sorted(os.listdir(input_path)) if f.endswith('.md')]
            # To avoid hitting rate limits or timeouts, we could process in batches
            # For this task, we'll process the first 20 docs as a representative sample
            for file in files[:5]:
                if file == ".gitkeep": continue
                
                fpath = os.path.join(input_path, file)
                print(f"  Analysing {file}...")
                doc_tree = asyncio.run(md_to_tree(
                    md_path=fpath,
                    if_add_node_summary='yes',
                    model=model
                ))
                self.tree['docs'].append(doc_tree)
        return self

    def save(self, path):
        print(f"Saving index to {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.tree, f, indent=4, ensure_ascii=False)

    def load(self, path):
        print(f"Loading index from {path}...")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.tree = json.load(f)
        return self

    def query(self, question, model="gpt-4o"):
        from .utils import ChatGPT_API
        
        print(f"Querying PageIndex with: {question}")
        
        # Step 1: Find relevant documents
        doc_summaries = []
        for i, doc in enumerate(self.tree.get('docs', [])):
            doc_name = doc.get('doc_name', f"Doc {i}")
            # Use the first node's summary as doc summary if needed
            summary = doc.get('structure', [{}])[0].get('summary', 'No summary')
            doc_summaries.append(f"Doc {i}: {doc_name} - {summary[:200]}...")
            
        summary_text = "\n".join(doc_summaries)
        select_prompt = f"""Given these documents, which ones (by index) might contain the answer to: "{question}"?
        {summary_text}
        Return a list of indices, e.g. [0, 2]. Limit to top 3.
        """
        
        selection_res = ChatGPT_API(model=model, prompt=select_prompt)
        try:
            import ast
            selected_indices = ast.literal_eval(selection_res.strip())
        except:
            selected_indices = [0] if self.tree['docs'] else []
            
        # Step 2: Retrieve from selected docs
        context_chunks = []
        for idx in selected_indices:
            if idx < len(self.tree['docs']):
                doc = self.tree['docs'][idx]
                for node in doc.get('structure', []):
                    # For simplicity, add all nodes from relevant documents
                    context_chunks.append(node.get('text', ''))
        
        context = "\n\n".join(context_chunks)[:10000] # Cap context
        
        # Step 3: Answer
        answer_prompt = f"""Context: {context}
        Question: {question}
        Answer based on context.
        """
        answer = ChatGPT_API(model=model, prompt=answer_prompt)
        
        return answer, context_chunks
