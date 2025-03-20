"""Bidirectional reference checker for documentation."""

from pathlib import Path
from typing import Dict, Set, List, Tuple
import networkx as nx
import sys
from datetime import datetime
from textwrap import indent

# Set project root path
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent

    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root path: {project_root}")
except Exception as e:
    # Fallback to current directory if path resolution fails
    sys.path.append(str(Path(__file__).parent))

from docs.scripts.xref.validators.reference_extractor import ReferenceExtractor


class BidirectionalChecker:
    """Checks bidirectional references in documentation."""
    
    def __init__(self):
        """Initialize the bidirectional checker."""
        self.extractor = ReferenceExtractor()
        self.graph = nx.DiGraph()
        self.valid_unidirectional_refs = {
            'CHANGELOG.md',
            'DOCS-CHANGELOG.md',
            'LICENSE',
            'README.md',
            'plan.md',
            'error_handling.md',
            'architecture.md'
        }
        
    def _resolve_reference(self, ref: str, source_file: Path) -> Path:
        """Improved reference resolution for Windows paths."""
        try:
            # Clean up the reference path
            ref = ref.strip('"\'')  # Remove any surrounding quotes
            
            if ref.startswith('http://') or ref.startswith('https://'):
                return None  # Skip external URLs
                
            if ref.startswith('/'):
                # Absolute reference from project root
                resolved = Path(ref.lstrip('/')).resolve()
            else:
                # Relative reference from source file
                resolved = (source_file.parent / ref).resolve()
                
            # Normalize path for Windows
            resolved = Path(str(resolved).replace('\\', '/'))
            
            # Verify the resolved path exists and is within the project
            if resolved.exists() and str(resolved).startswith(str(self.project_root)):
                return resolved
            return None
        except Exception as e:
            print(f"Error resolving reference {ref}: {str(e)}")
            return None
            
    def _validate_reference(self, source: Path, target: Path) -> bool:
        """Validate a reference against all rules."""
        # Check version consistency
        if not self._check_version_consistency(source, target):
            return False
            
        # Check anchor validation
        if '#' in str(target):
            if not self._validate_anchor(target):
                return False
                
        return True
        
    def _check_version_consistency(self, source: Path, target: Path) -> bool:
        """Check if referenced versions are consistent."""
        # Implementation would check version numbers in files
        return True  # Placeholder
        
    def _validate_anchor(self, target: Path) -> bool:
        """Validate that anchor references exist in target file."""
        # Implementation would check for anchor existence
        return True  # Placeholder
        
    def build_reference_graph(self, root_dir: str) -> nx.DiGraph:
        """Build graph with enhanced validation."""
        root_path = Path(root_dir)
        
        for file_path in self._get_doc_files(root_path):
            try:
                refs = self.extractor.extract_from_file(str(file_path))
                self.graph.add_node(str(file_path))
                
                for ref in refs:
                    target_path = self._resolve_reference(ref, file_path)
                    if target_path and self._validate_reference(file_path, target_path):
                        self.graph.add_edge(str(file_path), str(target_path))
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                
        return self.graph
        
    def _get_doc_files(self, root_dir: Path) -> List[Path]:
        """Get all documentation files in directory.
        
        Args:
            root_dir: Root directory to scan
            
        Returns:
            List of documentation file paths
        """
        doc_files = []
        for ext in ['.md', '.py', '.yaml', '.yml']:
            doc_files.extend(root_dir.rglob(f'*{ext}'))
        return doc_files
        
    def _is_valid_unidirectional(self, source: str, target: str) -> bool:
        """Check if a unidirectional reference is valid.
        
        Args:
            source: Source file path
            target: Target file path
            
        Returns:
            True if the reference is valid, False otherwise
        """
        target_name = Path(target).name
        return target_name in self.valid_unidirectional_refs
        
    def find_unidirectional_references(self) -> List[Tuple[str, str]]:
        """Find references that are not bidirectional.
        
        Returns:
            List of (source, target) pairs with invalid unidirectional references
        """
        unidirectional = []
        for source, target in self.graph.edges():
            if not self.graph.has_edge(target, source) and not self._is_valid_unidirectional(source, target):
                unidirectional.append((source, target))
        return unidirectional
        
    def find_orphaned_documents(self) -> Set[str]:
        """Improved orphaned document detection."""
        orphaned = set()
        valid_orphans = {
            'CHANGELOG.md', 'DOCS-CHANGELOG.md', 'LICENSE',
            'README.md', 'plan.md', 'error_handling.md'
        }
        
        for node in self.graph.nodes():
            node_name = Path(node).name
            if (self.graph.in_degree(node) == 0 and 
                node_name not in valid_orphans and
                node_name not in self.valid_unidirectional_refs):
                orphaned.add(node)
        return orphaned
        
    def generate_reference_report(self) -> Dict:
        """Generate a comprehensive reference report.
        
        Returns:
            Dictionary containing reference statistics and issues
        """
        unidirectional = self.find_unidirectional_references()
        orphaned = self.find_orphaned_documents()
        
        # Format unidirectional references
        formatted_unidirectional = "\n".join(
            f"- {source} → {target}" 
            for source, target in unidirectional
        )
        
        # Format orphaned documents
        formatted_orphaned = "\n".join(f"- {doc}" for doc in orphaned)
        
        # Format strongly connected components
        scc = list(nx.strongly_connected_components(self.graph))
        formatted_scc = "\n".join(
            f"- Component {i+1}: {', '.join(comp)}"
            for i, comp in enumerate(scc)
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_documents': self.graph.number_of_nodes(),
                'total_references': self.graph.number_of_edges(),
                'unidirectional_references': len(unidirectional),
                'orphaned_documents': len(orphaned),
                'average_references_per_doc': round(
                    self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
                    2
                )
            },
            'details': {
                'unidirectional_pairs': formatted_unidirectional,
                'orphaned_list': formatted_orphaned,
                'strongly_connected_components': formatted_scc
            }
        }
        
    def export_graph(self, output_path: str):
        """Export the reference graph to a visualization file.
        
        Args:
            output_path: Path to save the visualization
        """
        try:
            # Create a more readable graph for visualization
            viz_graph = nx.DiGraph()
            for node in self.graph.nodes():
                viz_graph.add_node(Path(node).name)
            for source, target in self.graph.edges():
                viz_graph.add_edge(Path(source).name, Path(target).name)
                
            # Export to DOT format
            nx.drawing.nx_pydot.write_dot(viz_graph, output_path)
            print(f"Graph successfully exported to {output_path}")
        except Exception as e:
            print(f"Error exporting graph: {str(e)}")
            
    def print_report(self):
        """Print formatted reference report to console."""
        report = self.generate_reference_report()
        
        print(f"\n=== Reference Validation Report ===")
        print(f"Generated at: {report['timestamp']}\n")
        
        print("Summary:")
        for key, value in report['summary'].items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
            
        print("\nDetails:")
        print("Unidirectional References:")
        print(indent(report['details']['unidirectional_pairs'], '  '))
        
        print("\nOrphaned Documents:")
        print(indent(report['details']['orphaned_list'], '  '))
        
        print("\nStrongly Connected Components:")
        print(indent(report['details']['strongly_connected_components'], '  '))
        
        print("\n=== End of Report ===\n")
        
    def validate_links(self, directory: str) -> Dict:
        """Validate all links in a directory.
        
        Args:
            directory: Directory to validate
            
        Returns:
            Validation report dictionary
        """
        self.build_reference_graph(directory)
        return self.generate_reference_report() 
    
if __name__ == "__main__":
    checker = BidirectionalChecker()
    report = checker.validate_links("docs")
    checker.print_report()

