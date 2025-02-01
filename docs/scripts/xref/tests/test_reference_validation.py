"""Tests for the cross-reference validation system."""

import unittest
from pathlib import Path
import tempfile
import shutil
import sys

# Add project root to Python path
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
from docs.scripts.xref.validators.bidirectional_checker import BidirectionalChecker

class TestReferenceValidation(unittest.TestCase):
    """Test cases for reference validation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary test directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        self.create_test_files()
        
        # Initialize validators
        self.extractor = ReferenceExtractor()
        self.checker = BidirectionalChecker()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def create_test_files(self):
        """Create test documentation files."""
        # Create test markdown file with references
        md_content = """
        # Test Document
        
        See [Reference 1](doc1.md) for more information.
        Also check [Reference 2](./doc2.md).
        
        ![Image](./images/test.png)
        """
        (self.test_dir / "test.md").write_text(md_content)
        
        # Create referenced files
        (self.test_dir / "doc1.md").write_text("# Document 1\n\nSee [Test](test.md)")
        (self.test_dir / "doc2.md").write_text("# Document 2\n\nSee [Test](./test.md)")
        
        # Create test Python file
        py_content = '''
        from docs.utils import helper
        from docs.models import model
        
        DOC_PATH = "docs/guide.md"
        '''
        (self.test_dir / "test.py").write_text(py_content)
        
    def test_markdown_reference_extraction(self):
        """Test extracting references from markdown files."""
        refs = self.extractor.extract_from_file(str(self.test_dir / "test.md"))
        
        self.assertIn("doc1.md", refs)
        self.assertIn("./doc2.md", refs)
        self.assertIn("./images/test.png", refs)
        
    def test_python_reference_extraction(self):
        """Test extracting references from Python files."""
        refs = self.extractor.extract_from_file(str(self.test_dir / "test.py"))
        
        self.assertIn("docs.utils", refs)
        self.assertIn("docs.models", refs)
        self.assertIn('"docs/guide.md"', refs)
        
    def test_bidirectional_validation(self):
        """Test bidirectional reference validation."""
        self.checker.build_reference_graph(str(self.test_dir))
        report = self.checker.generate_reference_report()
        
        self.assertEqual(report['total_documents'], 3)  # test.md, doc1.md, doc2.md
        self.assertTrue(report['total_references'] >= 4)  # At least 4 references
        
    def test_orphaned_documents(self):
        """Test finding orphaned documents."""
        # Create an orphaned document
        (self.test_dir / "orphaned.md").write_text("# Orphaned Document")
        
        self.checker.build_reference_graph(str(self.test_dir))
        orphaned = self.checker.find_orphaned_documents()
        
        self.assertGreaterEqual(len(orphaned), 1)
        self.assertTrue(any("orphaned.md" in doc for doc in orphaned))
        
    def test_unidirectional_references(self):
        """Test finding unidirectional references."""
        # Create a one-way reference
        (self.test_dir / "one_way.md").write_text("# One-way Document\n\nNo references.")
        (self.test_dir / "referrer.md").write_text("See [One-way](one_way.md)")
        
        self.checker.build_reference_graph(str(self.test_dir))
        unidirectional = self.checker.find_unidirectional_references()
        
        self.assertGreaterEqual(len(unidirectional), 1)
        self.assertTrue(any("one_way.md" in str(pair) for pair in unidirectional))
        
    def test_reference_resolution(self):
        """Test reference path resolution."""
        source_file = self.test_dir / "test.md"
        
        # Test relative reference
        relative_ref = "./doc1.md"
        resolved = self.checker._resolve_reference(relative_ref, source_file)
        self.assertEqual(resolved, (self.test_dir / "doc1.md").resolve())
        
        # Test absolute reference
        absolute_ref = "/doc2.md"
        resolved = self.checker._resolve_reference(absolute_ref, source_file)
        self.assertEqual(resolved, Path("doc2.md"))
        
if __name__ == '__main__':
    unittest.main() 