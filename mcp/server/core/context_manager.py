"""
Context manager for MCP server.
Manages file context, project structure, and semantic analysis.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path
import hashlib
from collections import OrderedDict

from ..utils.logger import default_logger as logger
from ..utils.mlflow_utils import default_mlflow as mlflow

class LRUCache:
    """LRU cache for file contents and context data."""
    
    def __init__(self, capacity: int = 100):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def remove(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear the cache."""
        self.cache.clear()

    def __len__(self) -> int:
        """Get cache size."""
        return len(self.cache)

class ContextManager:
    """
    Context manager for handling file context and project structure.
    Implements caching, semantic analysis, and MLflow integration.
    """
    
    def __init__(
        self,
        cache_size: int = 100,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        semantic_batch_size: int = 10
    ):
        """
        Initialize context manager.
        
        Args:
            cache_size: Maximum number of files to cache
            max_file_size: Maximum file size to process
            semantic_batch_size: Number of files to process in semantic batch
        """
        self.cache = LRUCache(cache_size)
        self.max_file_size = max_file_size
        self.semantic_batch_size = semantic_batch_size
        
        # Track file hashes for change detection
        self.file_hashes = {}
        
        # Statistics
        self.start_time = datetime.now()
        self.processed_files = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            "Context manager initialized",
            extra_fields={
                "cache_size": cache_size,
                "max_file_size": max_file_size
            }
        )

    async def get_context(self, file_path: str) -> Dict[str, Any]:
        """
        Get context information for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file context
        """
        try:
            # Check cache first
            cached = self.cache.get(file_path)
            if cached:
                self.cache_hits += 1
                logger.debug(
                    f"Cache hit for file: {file_path}",
                    extra_fields={"cache_hits": self.cache_hits}
                )
                return cached
            
            self.cache_misses += 1
            
            # Get file info
            file_info = await self._get_file_info(file_path)
            if not file_info:
                raise ValueError(f"Failed to get file info: {file_path}")
            
            # Get file content if size is reasonable
            content = None
            if file_info["size"] <= self.max_file_size:
                content = await self._read_file(file_path)
            
            # Build context
            context = {
                "file_info": file_info,
                "content": content,
                "semantic_info": await self._analyze_semantics(file_path, content),
                "related_files": await self._find_related_files(file_path),
                "mlflow_context": await self._get_mlflow_context(file_path)
            }
            
            # Cache the context
            self.cache.put(file_path, context)
            self.processed_files += 1
            
            logger.info(
                f"Generated context for file: {file_path}",
                extra_fields={
                    "context_size": len(str(context)),
                    "processed_files": self.processed_files
                }
            )
            
            return context
            
        except Exception as e:
            logger.error(
                f"Failed to get context: {str(e)}",
                extra_fields={"file_path": file_path}
            )
            raise

    async def get_structure(self) -> Dict[str, Any]:
        """
        Get the current project structure.
        
        Returns:
            Dictionary containing project structure
        """
        try:
            structure = {
                "timestamp": datetime.now().isoformat(),
                "root": await self._build_directory_tree(os.getcwd())
            }
            
            logger.info(
                "Generated project structure",
                extra_fields={"structure_size": len(str(structure))}
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to get structure: {str(e)}")
            raise

    async def refresh(self):
        """Force refresh of the context cache."""
        try:
            self.cache.clear()
            self.file_hashes.clear()
            logger.info("Context cache refreshed")
            
        except Exception as e:
            logger.error(f"Failed to refresh cache: {str(e)}")
            raise

    async def _get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get basic file information."""
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

    async def _read_file(self, file_path: str) -> Optional[str]:
        """Read file content."""
        try:
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Calculate file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            self.file_hashes[file_path] = file_hash
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            return None

    async def _analyze_semantics(
        self,
        file_path: str,
        content: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze file semantics."""
        try:
            if not content:
                return {}
                
            # Basic semantic analysis
            semantics = {
                "language": self._detect_language(file_path),
                "imports": self._extract_imports(content),
                "functions": self._extract_functions(content),
                "classes": self._extract_classes(content)
            }
            
            return semantics
            
        except Exception as e:
            logger.error(f"Failed to analyze semantics: {str(e)}")
            return {}

    async def _find_related_files(self, file_path: str) -> List[str]:
        """Find files related to the given file."""
        try:
            related = []
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            # Look for files with similar names
            directory = os.path.dirname(file_path)
            for entry in os.scandir(directory):
                if entry.is_file():
                    other_base = os.path.splitext(entry.name)[0]
                    if (other_base.startswith(base_name) or
                            base_name.startswith(other_base)):
                        related.append(entry.path)
            
            return related
            
        except Exception as e:
            logger.error(f"Failed to find related files: {str(e)}")
            return []

    async def _get_mlflow_context(self, file_path: str) -> Dict[str, Any]:
        """Get MLflow context for the file."""
        try:
            # Get relevant MLflow runs
            experiments = mlflow.list_experiments()
            context = {
                "experiments": experiments,
                "recent_runs": []  # Add logic to find relevant runs
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get MLflow context: {str(e)}")
            return {}

    async def _build_directory_tree(self, path: str) -> Dict[str, Any]:
        """Build directory tree structure."""
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
                    elif entry.is_dir():
                        tree["children"].append(
                            await self._build_directory_tree(entry.path)
                        )
            except PermissionError:
                tree["error"] = "Permission denied"
            
            return tree
            
        except Exception as e:
            logger.error(f"Failed to build directory tree: {str(e)}")
            return {"error": str(e)}

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension."""
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

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        return self._get_file_type(file_path)

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from content."""
        imports = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append(line)
        return imports

    def _extract_functions(self, content: str) -> List[str]:
        """Extract function definitions from content."""
        functions = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('def '):
                functions.append(line.split('(')[0][4:])
        return functions

    def _extract_classes(self, content: str) -> List[str]:
        """Extract class definitions from content."""
        classes = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('class '):
                classes.append(line.split('(')[0][6:])
        return classes

    def is_healthy(self) -> bool:
        """Check if context manager is healthy."""
        return True  # Add more sophisticated health checks

    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            "processed_files": self.processed_files,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache),
            "uptime": self.get_uptime()
        }