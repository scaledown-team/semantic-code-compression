# AST Heuristic Context Compression

## Hypotheses: 
 - We can maintain task reliability while compressing code by partitioning code into meaningful chunks using abstract syntax trees and summarizing or omitting irrelevant nodes and subtrees
 - Language servers can allow us to provide relevant context more effectively than dumping code, reducing context rot 

Target demonstration 60% compression and 95% reliability rates

## AST

This pipeline partitions source code into meaningful chunks using ASTs.

Experimentally developed heuristics and pruning algorithms are applied to the AST to determine relevance and omit or summarize nodes and subtrees.

## LSP

Language servers allow us to provide relevant context rather than repeatedly putting that burden on the model, taking the burden ff of the model to count references or idnetify errors, allowing it to still be effective with less code input.