"""Reference extractor for documentation cross-references."""

import re
from pathlib import Path
from typing import List, Dict, Set
import yaml
import markdown
from bs4 import BeautifulSoup

class ReferenceExtractor:
    """Extracts references from documentation files."""
    
    def __init__(self, config_path: str = "docs/scripts/xref/config.yaml"):
        """Initialize the reference extractor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.references = set()
        self.invalid_references = set()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "scan_directories": {
                "primary": ["docs/", "guides/"],
                "api": ["docs/api/"],
                "examples": ["docs/examples/"]
            },
            "reference_types": {
                "internal": [".md", ".py", ".yaml"],
                "api": [".swagger", ".openapi"],
                "code": [".py", ".json", ".yaml"]
            }
        }
        
    def extract_from_file(self, file_path: str) -> Set[str]:
        """Extract references from a single file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Set of references found in the file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        content = file_path.read_text(encoding='utf-8')
        refs = set()
        
        # Extract markdown links
        if file_path.suffix == '.md':
            refs.update(self._extract_markdown_refs(content))
            
        # Extract Python imports and references
        elif file_path.suffix == '.py':
            refs.update(self._extract_python_refs(content))
            
        # Extract YAML references
        elif file_path.suffix in ['.yaml', '.yml']:
            refs.update(self._extract_yaml_refs(content))
            
        return refs
        
    def _extract_markdown_refs(self, content: str) -> Set[str]:
        """Extract references from markdown content."""
        refs = set()
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all links
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith(('http://', 'https://')):
                refs.add(href)
                
        # Find all images
        for img in soup.find_all('img'):
            src = img.get('src')
            if src and not src.startswith(('http://', 'https://')):
                refs.add(src)
                
        return refs
        
    def _extract_python_refs(self, content: str) -> Set[str]:
        """Extract references from Python content."""
        refs = set()
        
        # Extract imports
        import_pattern = r'^(?:from|import)\s+([\w.]+)'
        refs.update(re.findall(import_pattern, content, re.MULTILINE))
        
        # Extract string references to docs
        doc_pattern = r'(?:["\']\/?docs\/[^\'"]+[\'"])'
        refs.update(re.findall(doc_pattern, content))
        
        return refs
        
    def _extract_yaml_refs(self, content: str) -> Set[str]:
        """Extract references from YAML content."""
        refs = set()
        try:
            data = yaml.safe_load(content)
            refs.update(self._find_refs_in_dict(data))
        except yaml.YAMLError:
            pass
        return refs
        
    def _find_refs_in_dict(self, data: Dict) -> Set[str]:
        """Recursively find references in dictionary."""
        refs = set()
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and (
                    value.endswith(tuple(self.config['reference_types']['internal'])) or
                    'docs/' in value or
                    'guides/' in value
                ):
                    refs.add(value)
                elif isinstance(value, (dict, list)):
                    refs.update(self._find_refs_in_dict(value))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    refs.update(self._find_refs_in_dict(item))
                elif isinstance(item, str) and (
                    item.endswith(tuple(self.config['reference_types']['internal'])) or
                    'docs/' in item or
                    'guides/' in item
                ):
                    refs.add(item)
        return refs
        
    def validate_references(self, refs: Set[str], base_path: str) -> Dict[str, List[str]]:
        """Validate extracted references.
        
        Args:
            refs: Set of references to validate
            base_path: Base path for resolving relative references
            
        Returns:
            Dictionary of valid and invalid references
        """
        base_path = Path(base_path)
        valid_refs = []
        invalid_refs = []
        
        for ref in refs:
            ref_path = base_path / ref.lstrip('/')
            if ref_path.exists():
                valid_refs.append(ref)
            else:
                invalid_refs.append(ref)
                
        return {
            'valid': valid_refs,
            'invalid': invalid_refs
        } 