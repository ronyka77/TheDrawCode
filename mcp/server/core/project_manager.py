"""
Project manager for MCP server.
Handles project structure analysis and file operations.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ..utils.logger import default_logger as logger

class ProjectManager:
    """
    Manages project structure and file operations.
    Provides methods for analyzing project structure and finding related files.
    """
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize project manager.
        
        Args:
            root_dir: Root directory of the project (defaults to current directory)
        """
        self.root_dir = root_dir or os.getcwd()
        self.start_time = datetime.now()
        self.processed_files = 0
        
        logger.info(
            "Project manager initialized",
            extra_fields={"root_dir": self.root_dir}
        )

    async def get_structure(self) -> Dict[str, Any]:
        """
        Get the current project structure.
        
        Returns:
            Dictionary containing project structure
        """
        try:
            structure = {
                "timestamp": datetime.now().isoformat(),
                "root": await self._build_directory_tree(self.root_dir)
            }
            
            logger.info(
                "Generated project structure",
                extra_fields={"structure_size": len(str(structure))}
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to get structure: {str(e)}")
            raise

    async def _build_directory_tree(self, path: str) -> Dict[str, Any]:
        """
        Build directory tree structure.
        
        Args:
            path: Directory path to analyze
            
        Returns:
            Dictionary containing directory tree
        """
        try:
            tree = {
                "name": os.path.basename(path) or path,
                "type": "directory",
                "children": []
            }
            
            try:
                entries = sorted(os.scandir(path), key=lambda e: e.name)
                for entry in entries:
                    if entry.is_file():
                        tree["children"].append({
                            "name": entry.name,
                            "type": "file",
                            "info": await self._get_file_info(entry.path)
                        })
                    elif entry.is_dir() and not self._should_ignore(entry.name):
                        tree["children"].append(
                            await self._build_directory_tree(entry.path)
                        )
            except PermissionError:
                tree["error"] = "Permission denied"
            
            return tree
            
        except Exception as e:
            logger.error(f"Failed to build directory tree: {str(e)}")
            return {"error": str(e)}

    async def _get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get basic file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            if not os.path.exists(file_path):
                return None
                
            stats = os.stat(file_path)
            return {
                "path": file_path,
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "type": self._get_file_type(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {str(e)}")
            return None

    def _get_file_type(self, file_path: str) -> str:
        """
        Determine file type from extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String indicating file type
        """
        ext = os.path.splitext(file_path)[1].lower()
        types = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".md": "markdown",
            ".txt": "text",
            ".yml": "yaml",
            ".yaml": "yaml"
        }
        return types.get(ext, "unknown")

    def _should_ignore(self, name: str) -> bool:
        """
        Check if a directory should be ignored.
        
        Args:
            name: Directory name
            
        Returns:
            Boolean indicating if directory should be ignored
        """
        ignore_patterns = {
            ".git",
            "__pycache__",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            ".venv",
            "venv",
            ".env"
        }
        return name in ignore_patterns

    def get_stats(self) -> Dict[str, Any]:
        """
        Get project manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "root_dir": self.root_dir,
            "processed_files": self.processed_files,
            "uptime": (datetime.now() - self.start_time).total_seconds()
        } 