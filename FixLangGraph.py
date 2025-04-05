# This file contains functions to update all integration files to use sets instead of lists for edges

def update_integration_file(file_path, find_replace_pairs):
    """
    Update a file with multiple find-replace operations

    Parameters:
    - file_path: Path to the file to update
    - find_replace_pairs: List of (find, replace) tuples
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply all replacements
    for find, replace in find_replace_pairs:
        content = content.replace(find, replace)

    # Write the updated content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Updated {file_path}")


def fix_all_integration_files():
    """Fix all LangGraph integration files to use sets for edges"""

    # Fix for cupcake_langgraph_integration.py
    update_integration_file(
        'cupcake_langgraph_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append(("classify", "process_emotions_for_dreamer"))',
             'new_edges.add(("classify", "process_emotions_for_dreamer"))'),
            ('new_edges.append(("process_emotions_for_dreamer", next_node))',
             'new_edges.add(("process_emotions_for_dreamer", next_node))'),
            ('new_edges.append(("reply", "cognitive_perspectives"))',
             'new_edges.add(("reply", "cognitive_perspectives"))'),
            ('new_edges.append(("cognitive_perspectives", next_node))',
             'new_edges.add(("cognitive_perspectives", next_node))'),
            ('new_edges.append(("dream", "enhanced_dream"))', 'new_edges.add(("dream", "enhanced_dream"))'),
            ('new_edges.append(("enhanced_dream", next_node))', 'new_edges.add(("enhanced_dream", next_node))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    # Fix for entity_relationship_integration.py
    update_integration_file(
        'entity_relationship_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append((edge[0], "identify_entities"))', 'new_edges.add((edge[0], "identify_entities"))'),
            ('new_edges.append(("identify_entities", "enhance_response_with_relationships"))',
             'new_edges.add(("identify_entities", "enhance_response_with_relationships"))'),
            ('new_edges.append(("enhance_response_with_relationships", "analyze_relationship_patterns"))',
             'new_edges.add(("enhance_response_with_relationships", "analyze_relationship_patterns"))'),
            ('new_edges.append(("analyze_relationship_patterns", next_node))',
             'new_edges.add(("analyze_relationship_patterns", next_node))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    # Fix for memory_integration.py
    update_integration_file(
        'memory_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append((incoming_edge, "retrieve_enhanced_memories"))',
             'new_edges.add((incoming_edge, "retrieve_enhanced_memories"))'),
            ('new_edges.append(("retrieve_enhanced_memories", "memory_cluster_analysis"))',
             'new_edges.add(("retrieve_enhanced_memories", "memory_cluster_analysis"))'),
            ('new_edges.append(("memory_cluster_analysis", outgoing_edge))',
             'new_edges.add(("memory_cluster_analysis", outgoing_edge))'),
            ('new_edges.append(("reply", "add_enhanced_memory"))', 'new_edges.add(("reply", "add_enhanced_memory"))'),
            ('new_edges.append(("add_enhanced_memory", next_node))',
             'new_edges.add(("add_enhanced_memory", next_node))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    # Fix for perception_integration.py
    update_integration_file(
        'perception_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append((incoming_edge, "enhanced_perception"))',
             'new_edges.add((incoming_edge, "enhanced_perception"))'),
            ('new_edges.append(("enhanced_perception", "perception_reflection"))',
             'new_edges.add(("enhanced_perception", "perception_reflection"))'),
            ('new_edges.append(("perception_reflection", outgoing_edge))',
             'new_edges.add(("perception_reflection", outgoing_edge))'),
            ('new_edges.append(("classify", "enhanced_perception"))',
             'new_edges.add(("classify", "enhanced_perception"))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    # Fix for narrative_integration.py
    update_integration_file(
        'narrative_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append((edge[0], "process_narrative"))', 'new_edges.add((edge[0], "process_narrative"))'),
            ('new_edges.append(("process_narrative", "reflect_on_narrative"))',
             'new_edges.add(("process_narrative", "reflect_on_narrative"))'),
            ('new_edges.append(("reflect_on_narrative", "update_self_narrative"))',
             'new_edges.add(("reflect_on_narrative", "update_self_narrative"))'),
            ('new_edges.append(("update_self_narrative", next_node))',
             'new_edges.add(("update_self_narrative", next_node))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    # Fix for entropic_identity_integration.py
    update_integration_file(
        'entropic_identity_integration.py',
        [
            # Change how we create the new edges
            ('new_edges = []', 'new_edges = set()'),
            # Change how we add to the new edges
            ('new_edges.append((edge[0], "identity_entropy_update"))',
             'new_edges.add((edge[0], "identity_entropy_update"))'),
            ('new_edges.append(("identity_entropy_update", "identity_emergence_check"))',
             'new_edges.add(("identity_entropy_update", "identity_emergence_check"))'),
            ('new_edges.append(("identity_emergence_check", next_node))',
             'new_edges.add(("identity_emergence_check", next_node))'),
            ('new_edges.append((edge[0], "enhanced_identity_prompt"))',
             'new_edges.add((edge[0], "enhanced_identity_prompt"))'),
            ('new_edges.append(("enhanced_identity_prompt", next_node))',
             'new_edges.add(("enhanced_identity_prompt", next_node))'),
            ('new_edges.append(edge)', 'new_edges.add(edge)'),
        ]
    )

    print("All integration files updated to use sets instead of lists for edges.")


if __name__ == "__main__":
    fix_all_integration_files()