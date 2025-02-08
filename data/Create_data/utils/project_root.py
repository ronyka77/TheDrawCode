from pathlib import Path
import os
import sys

def setup_project_root() -> Path:
    """
    Determines and sets the project root, adding it to sys.path if not already present.
    Handles network path resolution elegantly.
    """
    try:
        # Determine the project root: assumes this file is inside a subfolder of the root.
        project_root = Path(__file__).parent.parent.resolve()
        if not project_root.exists():
            # If the path does not exist (e.g., due to network path issues),
            # reformat the path using raw strings.
            project_root = Path(r"\\".join(str(project_root).split("\\")))
        
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.append(project_root_str)
            print(f"Added project root to sys.path: {project_root_str}")
        else:
            print(f"Project root already in sys.path: {project_root_str}")
        

        return project_root
    
    except Exception as e:
        print(f"Error setting project root path: {e}", exc_info=True)
        # Fallback: use the parent of the current working directory.
        fallback = Path(os.path.dirname(os.getcwd())).resolve()
        fallback_str = str(fallback)
        if fallback_str not in sys.path:
            sys.path.append(fallback_str)
            print(f"Used fallback project root: {fallback_str}")
        return fallback 