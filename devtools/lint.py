"""
Convenience script to run all linters.
"""

import subprocess
import sys


def run_command(command: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)  # Show output directly
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result.returncode


def main() -> int:
    """Run all linters."""
    exit_code = 0

    print("\n--- Running Ruff Check ---")
    # Add --fix if you want to automatically fix issues
    exit_code |= run_command([sys.executable, "-m", "ruff", "check", ".", "--show-fixes"])

    print("\n--- Running Ruff Format ---")
    exit_code |= run_command([sys.executable, "-m", "ruff", "format", "."])

    # Add other linters/checks here if needed, e.g.:
    # print("\n--- Running BasedPyright ---")
    # exit_code |= run_command([sys.executable, "-m", "basedpyright", "."])

    if exit_code == 0:
        print("\nLinting passed successfully!")
    else:
        print("\nLinting failed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
