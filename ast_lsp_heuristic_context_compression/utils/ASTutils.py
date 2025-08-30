import ast
from typing import Optional, Any
class ASTManipulation(ast.NodeTransformer):
  """Handles the core challenge of maintaining valid AST structures"""
  def __init__(self,target_node:ast.AST,operation:str,summary_text:Optional[str]=None):
    self.target_node=target_node
    self.operation=operation
    self.summary_text=summary_text
    self.target_found=False
  def visit(self,node):
    if id(node)==id(self.target_node):
      self.target_found=True
      if self.operation=="omit":
        return None
      elif self.operation in ['replace','replace_subtree']:
        return ast.Expr(value=ast.Constant(value=self.summary_text))
    return self.generic_visit(node)

def replace_node_with_summary(tree:ast.AST,target_node:ast.AST,summary_text:str)->ast.AST:
  """
  Replaces a target node with a summary string literal.
  """
  if not isinstance(target_node,ast.stmt):
    raise ValueError("target_node must be a statement (ast.stmt)")
  manipulator=ASTManipulation(target_node,'replace',summary_text)
  new_tree=manipulator.visit(tree)

  if not manipulator.target_found:
    raise ValueError("target_node not found in the AST tree")
  return ast.fix_missing_locations(new_tree)
def omit_node(tree:ast.AST,target_node:ast.AST)->ast.AST:
  "Completely remove a target node from the AST"
  manipulator=ASTManipulation(target_node,'omit')
  new_tree=manipulator.visit(tree)

  if not manipulator.target_found:
    raise ValueError("target_node not found in the AST tree")
  return ast.fix_missing_locations(new_tree)
def replace_subtree_with_summary(tree:ast.AST,subtree_root_node:ast.AST,summary_text:str)->ast.AST:
  """
  Replace an entire subtree with a summary.
  """
  manipulator=ASTManipulation(subtree_root_node,'replace_subtree',summary_text)
  new_tree=manipulator.visit(tree)
  if not manipulator.target_found:
    raise ValueError("subtree_root_node not found in the AST tree")
  return ast.fix_missing_locations(new_tree)
def find_nodes_by_type(tree: ast.AST, node_type: type) -> list[ast.AST]:
    """Find all nodes of a specific type in the AST."""
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            nodes.append(node)
    return nodes
def find_function_by_name(tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
    """Find a function definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None
def find_class_by_name(tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
    """Find a class definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None
def validate_ast(tree: ast.AST) -> bool:
    """
    Validate that an AST can be successfully compiled and unparsed.
    """
    try:
        compile(tree, filename="<ast>", mode="exec")

        ast.unparse(tree)

        return True
    except Exception:
        return False