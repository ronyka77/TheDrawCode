#!/usr/bin/env python3
"""
Script to delete .log files that are 5KB or smaller from the logs folder.
"""

import pathlib


def delete_small_log_files(logs_dir: str = "logs", max_size_kb: int = 5) -> None:
    """
    Delete .log files that are smaller than or equal to the specified size.

    Args:
        logs_dir: Path to the logs directory (default: "logs")
        max_size_kb: Maximum file size in KB to delete (default: 5)
    """
    logs_path = pathlib.Path(logs_dir)

    # Check if logs directory exists
    if not logs_path.exists():
        print(f"❌ Logs directory '{logs_dir}' does not exist.")
        return

    if not logs_path.is_dir():
        print(f"❌ '{logs_dir}' is not a directory.")
        return

    max_size_bytes = max_size_kb * 1024  # Convert KB to bytes
    deleted_files: list[str] = []
    skipped_files: list[str] = []

    print(
        f"🔍 Scanning for .log files in '{logs_dir}' and subfolders that are {max_size_kb}KB or smaller..."
    )

    # Find all .log files recursively (including subfolders)
    log_files = list(logs_path.rglob("*.log"))

    if not log_files:
        print(f"ℹ️  No .log files found in '{logs_dir}'")
        return

    print(f"📁 Found {len(log_files)} .log file(s)")

    for log_file in log_files:
        try:
            file_size = log_file.stat().st_size
            file_size_kb = file_size / 1024

            if file_size <= max_size_bytes:
                log_file.unlink()  # Delete the file
                relative_path = log_file.relative_to(logs_path)
                deleted_files.append(f"{relative_path} ({file_size_kb:.2f}KB)")
                print(f"🗑️  Deleted: {relative_path} ({file_size_kb:.2f}KB)")
            else:
                relative_path = log_file.relative_to(logs_path)
                skipped_files.append(f"{relative_path} ({file_size_kb:.2f}KB)")
                print(
                    f"⏭️  Skipped: {relative_path} ({file_size_kb:.2f}KB) - larger than {max_size_kb}KB"
                )

        except OSError as e:
            relative_path = log_file.relative_to(logs_path)
            print(f"❌ Error processing {relative_path}: {e}")

    # Summary
    print("\n📊 Summary:")
    print(f"   • Deleted: {len(deleted_files)} file(s)")
    print(f"   • Skipped: {len(skipped_files)} file(s)")

    if deleted_files:
        print("\n🗑️  Deleted files:")
        for file_info in deleted_files:
            print(f"   • {file_info}")


def main():
    """Main function with safety confirmation."""
    logs_dir = "logs"
    max_size_kb = 5

    print(
        f"⚠️  This script will delete .log files {max_size_kb}KB or smaller from '{logs_dir}' folder."
    )

    # Safety confirmation
    try:
        confirm = input("Do you want to continue? (y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("❌ Operation cancelled.")
            return
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
        return

    delete_small_log_files(logs_dir, max_size_kb)


if __name__ == "__main__":
    main()
