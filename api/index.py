"""
Vercel Serverless Function entry point.
Exposes the FastAPI application for Vercel deployment.
"""
import os
import sys

# Add root directory to sys.path so modules in src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.api.server import app
