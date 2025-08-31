import os
import ast

def extract_semantic_units(file_path):
    """Extract file, classes, and functions from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    semantic_units = []

    # File as a whole
    semantic_units.append({
        "module": os.path.splitext(os.path.basename(file_path))[0],
        "class": None,
        "file_name": os.path.basename(file_path),
        "semantic_unit": "file",
        "code": source
    })

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_code = ast.get_source_segment(source, node)
            semantic_units.append({
                "module": os.path.splitext(os.path.basename(file_path))[0],
                "class": node.name,
                "file_name": os.path.basename(file_path),
                "semantic_unit": "class",
                "code": class_code
            })
        elif isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(source, node)
            semantic_units.append({
                "module": os.path.splitext(os.path.basename(file_path))[0],
                "class": None,
                "file_name": os.path.basename(file_path),
                "semantic_unit": "function",
                "code": func_code
            })
    return semantic_units