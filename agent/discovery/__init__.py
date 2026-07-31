"""Discovery: scan, filter, stack detection, dependency graph, snippets, explore."""

from agent.discovery.scan import scan_file_tree
from agent.discovery.heuristics import filter_candidates, rank_by_task_signals
from agent.discovery.stack import detect_stack
from agent.discovery.graph import (
    build_dependency_graph,
    neighbors,
    topological_edit_order,
)
from agent.discovery.snippets import read_line_numbered, excerpt_for_prompt
from agent.discovery.explore import explore_with_llm

__all__ = [
    "scan_file_tree",
    "filter_candidates",
    "rank_by_task_signals",
    "detect_stack",
    "build_dependency_graph",
    "neighbors",
    "topological_edit_order",
    "read_line_numbered",
    "excerpt_for_prompt",
    "explore_with_llm",
]
