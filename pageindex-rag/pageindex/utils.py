import tiktoken
import openai
import logging
import os
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
import yaml
from pathlib import Path
from types import SimpleNamespace as config

load_dotenv()
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

def count_tokens(text, model="gpt-4o"):
    if not text:
        return 0
    enc = tiktoken.encoding_for_model(model if model else "gpt-4o")
    tokens = enc.encode(text)
    return len(tokens)

def ChatGPT_API_with_finish_reason(model, prompt, api_key=None, chat_history=None):
    if not api_key: api_key = CHATGPT_API_KEY
    max_retries = 10
    client = openai.OpenAI(api_key=api_key)
    for i in range(max_retries):
        try:
            messages = chat_history.copy() if chat_history else []
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            if response.choices[0].finish_reason == "length":
                return response.choices[0].message.content, "max_output_reached"
            else:
                return response.choices[0].message.content, "finished"
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                return "Error", "failed"

def ChatGPT_API(model, prompt, api_key=None, chat_history=None):
    if not api_key: api_key = CHATGPT_API_KEY
    res, _ = ChatGPT_API_with_finish_reason(model, prompt, api_key, chat_history)
    return res

async def ChatGPT_API_async(model, prompt, api_key=None):
    if not api_key: api_key = CHATGPT_API_KEY
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            async with openai.AsyncOpenAI(api_key=api_key) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content
        except Exception as e:
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                return "Error"

def extract_json(content):
    try:
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()
        json_content = json_content.replace('None', 'null')
        return json.loads(json_content)
    except:
        return {}

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        if 'nodes' in data:
            node_id = write_node_id(data['nodes'], node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id

def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = [structure]
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

async def generate_node_summary(node, model=None):
    prompt = f"Summarize this part of the document: {node.get('text', '')}"
    return await ChatGPT_API_async(model, prompt)

async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure

def format_structure(structure, order=None):
    return structure # Simplified

class ConfigLoader:
    def load(self, user_opt=None):
        return config(**(user_opt or {}))
