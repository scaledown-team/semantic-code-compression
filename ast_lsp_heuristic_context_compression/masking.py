import ast
import random

class MaskableFinder(ast.NodeVisitor):
    """
    Traverses an AST to find maskable elements (attribute accesses and method calls),
    excluding those within __init__ methods.
    """
    def __init__(self):
        self.maskable_elements = []
        self._in_class = False
        self._current_method = None

    def visit_ClassDef(self, node):
        self._in_class = True
        self.generic_visit(node)
        self._in_class = False
        self._current_method = None

    def visit_FunctionDef(self, node):
        if self._in_class:
            self._current_method = node.name
        self.generic_visit(node)
        if self._in_class:
            self._current_method = None

    def visit_Call(self, node):
        # Exclude masking inside __init__ or other specific methods if needed
        if self._in_class and self._current_method == '__init__':
             self.generic_visit(node)
             return

        if isinstance(node.func, ast.Attribute):
            self.maskable_elements.append(node)
            # Visit arguments and keywords individually to avoid generic_visit on a list
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword)
            return

        self.generic_visit(node)


    def visit_Attribute(self, node):
        # Exclude masking inside __init__ or other specific methods if needed
        if self._in_class and self._current_method == '__init__':
            self.generic_visit(node)
            return

        self.maskable_elements.append(node)
        self.generic_visit(node)

class MaskingTransformer(ast.NodeTransformer):
    """
    Transforms an AST by replacing selected attribute accesses and method calls
    with unique identifiers and generates a mapping.
    """
    def __init__(self, elements_to_mask, code_text):
        self.elements_to_mask = elements_to_mask
        self.code_text = code_text
        self.mask_counter = 0
        self.mapping = {}

    def visit(self, node):
        original_node = node

        if node in self.elements_to_mask:
            mask_id = f"MASK_{self.mask_counter}"

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # Masking method calls: store the method name
                original_snippet = node.func.attr
                new_func = ast.Attribute(value=node.func.value, attr=mask_id, ctx=node.func.ctx)
                new_node = ast.Call(func=new_func, args=node.args, keywords=node.keywords)

            elif isinstance(node, ast.Attribute):
                # Masking attribute accesses: store the attribute name
                original_snippet = node.attr
                new_node = ast.Attribute(value=node.value, attr=mask_id, ctx=node.ctx)
            else:
                 # Should not happen with current MaskableFinder logic, but as a fallback
                 original_snippet = ast.get_source_segment(self.code_text, original_node) or "Unknown Snippet"
                 new_node = ast.Name(id=mask_id, ctx=ast.Load())

            self.mapping[mask_id] = original_snippet
            self.mask_counter += 1

            # Preserve line number and column offset if possible for better debugging
            if hasattr(original_node, 'lineno') and hasattr(original_node, 'col_offset'):
                new_node.lineno = original_node.lineno
                new_node.col_offset = original_node.col_offset

            return ast.fix_missing_locations(new_node)

        return self.generic_visit(node)

def mask_code(code_text: str, masking_rate: float = 0.05):
    """
    Masks attribute accesses and method calls in the provided code text.

    Args:
        code_text: A string containing the Python code to be masked.
        masking_rate: The approximate rate of masking to apply to attribute accesses
                      and method calls. Defaults to 0.05.

    Returns:
        A tuple containing:
            - masked_code: The code text with selected elements replaced by unique identifiers.
            - mapping: A dictionary mapping the unique identifiers to their original statements.
    """
    tree = ast.parse(code_text)
    finder = MaskableFinder()
    finder.visit(tree)
    maskable_elements = finder.maskable_elements

    num_to_mask = int(len(maskable_elements) * masking_rate)
    elements_to_mask = random.sample(maskable_elements, min(num_to_mask, len(maskable_elements)))

    transformer = MaskingTransformer(elements_to_mask, code_text)
    masked_tree = transformer.visit(tree)
    mapping = transformer.mapping

    masked_code_text = ast.unparse(masked_tree)

    return masked_code_text, mapping

