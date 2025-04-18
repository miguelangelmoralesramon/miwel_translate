from setuptools import setup, find_packages

setup(
    name="miwel-translate",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "python-multipart>=0.0.6",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="English to Spanish translator using a custom Transformer model",
    keywords="translation, transformer, nlp, fastapi",
    url="https://github.com/yourusername/miwel-translate",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)