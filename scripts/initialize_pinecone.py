import os
from os.path import dirname
from flask import Flask, request, jsonify
from pinecone import Pinecone
from transformers import AutoTokenizer,AutoModel,BertTokenizer, BertModel
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get('PINECONE_API_KEY')

pc = Pinecone(api_key=API_KEY)

index_name = 'its'

index = pc.Index(index_name)
print(f"Connected to index '{index_name}'")
