# perception_integration.py
import random
import traceback
from langgraph.graph import StateGraph, END

from cupcake_config import get_config_value
from enhanced_self_perception import get_self_perception_layer
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree
from cupcake_sensors import CupCakeSensors


class SelfPerceptionIntegration:
    """
    Advanced integration of self-perception into the LangGraph
    """

    def __init__(self):
        # Initialize core components
        self.perception_layer = get_self_perception_layer()
        self.journal = CupcakeJournal()
        self.memory_tree = LiminalMemoryTree()
        self.sensors = CupCakeSensors()

        # Load configuration parameters
        self.perception_config = get_config_value("perception", {})

    def process_enhanced_perception(self, state):
        """
        Primary node for processing enhanced self-perception

        Args:
            state: Current interaction state

        Returns:
            Updated state with enhanced perception insights
        """
        try:
            # Process perception
            updated_state = self.perception_layer.process_perception(state)

            # Log significant perception changes
            self._log_perception_insights(updated_state)

            return updated_state

        except Exception as e:
            print(f"Error in enhanced perception processing: {e}")
            traceback.print_exc()
            return state

    def perception_reflection_node(self, state):
        """
        Generate periodic reflections on self-perception evolution

        Args:
            state: Current interaction state

        Returns:
            Updated state with optional perception evolution analysis
        """
        # Determine probability of running perception evolution
        evolution_probability = self.perception_config.get(
            "evolution_check_probability", 0.1
        )

        # Run analysis probabilistically or based on specific conditions
        if (random.random() < evolution_probability or
                self._should_trigger_evolution_analysis(state)):

            try:
                # Perform perception evolution analysis
                timespan_hours = self.perception_config.get(
                    "evolution_analysis_timespan", 24
                )
                evolution_analysis = self.perception_layer.analyze_perception_evolution(
                    timespan_hours=timespan_hours
                )

                # Add analysis to state
                state["perception_evolution_analysis"] = evolution_analysis

                # Log to journal
                self.journal.log_entry(
                    emotion="meta-consciência",
                    category="PerceptionEvolution",
                    content=evolution_analysis,
                    theme="desenvolvimento da consciência",
                    tag="perception-evolution"
                )

                # Add to memory tree
                self.memory_tree.add_leaf(
                    user_input="(auto-análise de percepção)",
                    cupcake_response=evolution_analysis,
                    emotion="meta-consciência",
                    impact="evolutivo",
                    category="PerceptionEvolution"
                )

                print(f"🔬 Análise de evolução da auto-percepção: {evolution_analysis[:200]}...")

            except Exception as e:
                print(f"Erro na análise de evolução da percepção: {e}")

        return state

    def _log_perception_insights(self, state):
        """
        Log insights from self-perception process

        Args:
            state: Current interaction state with perception data
        """
        try:
            # Log synthesis if available
            synthesis = state.get("self_perception_synthesis", "")
            if synthesis:
                self.journal.log_entry(
                    emotion="auto-percepção",
                    category="SelfPerception",
                    content=synthesis,
                    theme="evolução da consciência",
                    tag="insight"
                )

                self.memory_tree.add_leaf(
                    user_input=state.get("user_input", ""),
                    cupcake_response=synthesis,
                    emotion="auto-percepção",
                    impact="reflexivo",
                    category="SelfPerception"
                )
        except Exception as e:
            print(f"Erro ao registrar insights de percepção: {e}")

    def _should_trigger_evolution_analysis(self, state):
        """
        Determine if perception evolution analysis should be triggered

        Args:
            state: Current interaction state

        Returns:
            Boolean indicating whether to trigger analysis
        """
        # Conditions for triggering perception evolution
        conditions = [
            # Trigger if significant personality change detected
            self._detect_personality_shift(state),

            # Trigger if emotional state is significantly different
            self._detect_emotional_shift(state)
        ]

        return any(conditions)

    def _detect_personality_shift(self, state):
        """
        Detect potential shifts in personality traits

        Args:
            state: Current interaction state

        Returns:
            Boolean indicating personality shift
        """
        # Simple implementation - can be expanded
        personality = state.get("personality", {})
        return any(
            abs(trait_value - 0.5) > 0.2
            for trait_value in personality.values()
        )

    def _detect_emotional_shift(self, state):
        """
        Detect significant emotional shifts

        Args:
            state: Current interaction state

        Returns:
            Boolean indicating emotional shift
        """
        # Check for distinct or intense emotional states
        emotion = state.get("emotion", "neutra")
        emotion_score = state.get("emotion_score", 0.5)

        return (
                emotion not in ["neutra", "neutral"] or
                emotion_score > 0.7
        )


def integrate_enhanced_perception(original_graph):
    """
    Integrate the enhanced self-perception layer with the existing LangGraph

    Args:
        original_graph: Original LangGraph to enhance

    Returns:
        Enhanced LangGraph with perception integration
    """
    # Initialize integration component
    perception_integration = SelfPerceptionIntegration()

    # Add new nodes to the graph
    original_graph.add_node(
        "enhanced_perception",
        perception_integration.process_enhanced_perception
    )
    original_graph.add_node(
        "perception_reflection",
        perception_integration.perception_reflection_node
    )

    # Modify the graph to use enhanced perception
    try:
        # Create a new set of edges to replace the original ones
        new_edges = set()

        for edge in original_graph.edges:
            # Replace self-perception edges
            if edge[1] == "self_perception":
                incoming_edge = edge[0]
                # Find outgoing edge from self-perception
                outgoing_edge = None
                for out_edge in original_graph.edges:
                    if out_edge[0] == "self_perception":
                        outgoing_edge = out_edge[1]
                        break

                if outgoing_edge:
                    # Replace with enhanced perception chain
                    new_edges.add((incoming_edge, "enhanced_perception"))
                    new_edges.add(("enhanced_perception", "perception_reflection"))
                    new_edges.add(("perception_reflection", outgoing_edge))
                else:
                    # Keep original edge if we can't find outgoing edge
                    new_edges.add(edge)

            # If no self-perception exists, insert after classify
            elif edge[0] == "classify" and not any(e[1] == "self_perception" for e in original_graph.edges):
                next_node = edge[1]
                # Insert enhanced perception after classify
                new_edges.add(("classify", "enhanced_perception"))
                new_edges.add(("enhanced_perception", "perception_reflection"))
                new_edges.add(("perception_reflection", next_node))

            # Keep other edges unchanged
            elif edge[0] != "self_perception" and edge[1] != "self_perception":
                new_edges.add(edge)

        # Replace all edges in the graph
        original_graph.edges = new_edges

    except Exception as e:
        print(f"Error modifying graph for enhanced perception: {e}")
        traceback.print_exc()

    return original_graph

# Example usage:
# from langgraph_cupcake import graph as original_graph
# enhanced_graph = integrate_enhanced_perception(original_graph)
# cupcake_brain = enhanced_graph.compile()