from setuptools import setup, find_packages

setup(
    name="mcp",
    version="0.1.0",
    description="Model Context Protocol Server",
    author="MCP Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.68.0,<0.69.0",
        "uvicorn>=0.15.0,<0.16.0",
        "pydantic>=1.8.0,<2.0.0",
        "python-multipart>=0.0.5,<0.1.0",
        "aiofiles>=0.7.0,<0.8.0",
        "watchdog>=2.1.0,<3.0.0",
        "mlflow>=1.20.0,<2.0.0",
        "websockets>=10.0,<11.0",
        "python-json-logger>=2.0.0,<3.0.0",
        "structlog>=21.1.0,<22.0.0",
        "sse-starlette>=1.6.1",
    ],
    python_requires=">=3.8",
    zip_safe=False
) 