"""Installation script for MCP Server."""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}':")
        print(e.stderr)
        return False

def main():
    """Main installation function."""
    print("Installing MCP Server...")
    
    # Ensure pip is up to date
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        sys.exit(1)
    
    # Install FastMCP SDK first
    if not run_command(f"{sys.executable} -m pip install fastmcp"):
        print("Failed to install FastMCP SDK")
        sys.exit(1)
    
    # Install other requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        print("Failed to install requirements")
        sys.exit(1)
    
    # Install package in development mode
    if not run_command(f"{sys.executable} -m pip install -e ."):
        print("Failed to install package")
        sys.exit(1)
    
    print("\nInstallation completed successfully!")
    print("\nYou can now run the server with:")
    print("  python -m mcp.server")
    print("  # or")
    print("  mcp-server")

if __name__ == "__main__":
    main() 