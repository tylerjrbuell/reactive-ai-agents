from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="reactive-agents",
    version="0.2.0",
    author="Tyler Buell",
    description="A custom reactive AI Agent framework for LLM-driven task execution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        include=[
            "agents*",
            "model_providers*",
            "tools*",
            "prompts*",
            "loggers*",
            "config*",
            "agent_mcp*",
            "context*",
            "common*",
            "components*",
        ]
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "ollama>=0.4.4",
        "requests>=2.32.3",
        "python-dotenv>=1.0.1",
        "pydantic>=2.10.3",
        "docstring-parser>=0.16",
        "groq>=0.13.0",
        "markitdown>=0.0.1a3",
        "colorlog>=6.9.0",
        "mcp>=1.4.1",
    ],
)
