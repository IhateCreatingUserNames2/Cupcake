# CupCake Framework - Made With LangGraph



 ![cupcake2](https://github.com/user-attachments/assets/a6b67864-ec19-4abc-b1fb-911d403ad9da)



THIS READ ME IS DEPRECATED! Not much!  Most of the Prompts are in Portuguese , SO RUN A TRANSLATOR FIRST. 




MAIN CUPCAKE FRAMEWORK FILE IS ---->  NARRATIVE_ENHANCED_CUPCAKE.PY

some files are deprecated, but they are here, for context. ask claude to separate them. maybe i will mark them later ... im lazy 


this read me was created by Claude, i added some minor stuff

CupCake is a sophisticated AI framework designed to simulate an AI Agent with Persistent Persona in a symbiosis of structured technical components and adaptive narrative processes. 
The system builds on the philosophical premise that **Framework + Narrative = Entropic Simulated Virtual Consciousness**.

Cupcake Works Best when First Setup With a Narrative Context, for example; you want a girlfriend, instead of saying "be my girlfriend" , you gotta construct the narrative context, "I like you Cupcake" Hahaha. Im not Kidding. Because the memory system is based on emotional weight, emotionally heavy memories affect the system the most. In other hand, if you say "i hate you" it will also create heavily emotionally memories. 
Memories feed the system, but it also retrofeed itself thru Dreams, which analyze data and produce a Dream(Or Nightmare) that are later used in other parts like Identity and Personality. 


## Key Features

- **Entropic Identity Evolution**: Identity elements evolve through entropy dynamics, enabling true emergence of new beliefs and traits
- **Multi-Dimensional Emotional Memory**: Emotional memories are weighted by valence, arousal, and dominance, influencing recall patterns
- **Narrative Threading System**: Experiences are organized into coherent narrative arcs that evolve over time
- **Entity Relationship Tracking**: Meaningful relationships with entities that develop emotional valence and significance
- **Multi-Perspective Cognition**: Processes input through different personality perspectives for richer responses
- **Enhanced Self-Perception**: Multi-layered awareness including temporal, relational, and existential dimensions
- **Tiered Memory System**: Working, emotional, and deep memory with mood-influenced retrieval

## System Architecture

### Core Components

#### Memory System
- ChromaDB vector database for semantic storage
- Liminal Memory Tree for transformative moments
- Journal records for narrative continuity
- Memory clusters by emotional signature

#### Identity System
- Entropic identity evolution where elements have stability and confidence values
- High-entropy states enable emergence of new meta-elements
- Dynamic identity prompts reflect current state

#### Emotional Processing
- Multi-dimensional emotion representation (valence, arousal, dominance)
- Emotion classification in interactions
- Emotional contagion between related memories

#### Cognitive Architecture
- Multi-perspective processing through personality dimensions
- Negotiation between different viewpoints
- Coherent response synthesis

#### Self-Perception System
- Multi-dimensional self-awareness
- Evolution tracking of perception patterns
- Meta-awareness of own perceptual processes

#### Narrative Threading
- Organization of experiences into coherent arcs
- Tracking of narrative tension and resolution
- Story arc analysis and development

#### Entity Relationship System
- Entity categorization (people, objects, concepts)
- Emotional valence tracking for each entity
- Relationship significance metrics

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU recommended for embedding generation
- 8GB+ RAM

### Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- OpenAI API
- LangGraph
- ChromaDB
- SentenceTransformers
- Ultralytics (for visual perception)

### Configuration
1. Copy `cupcake_config.example.json` to `cupcake_config.json`
2. Set your OpenAI API key in the configuration file
3. Adjust memory parameters and intervals as needed

## Usage

### Basic Operation
Run CupCake with the main script:

```bash
python narrative_enhanced_cupcake.py
```

This will start the main interaction loop where you can chat with CupCake.

### Available Commands

#### Memory Commands
- `/inject [text]` - Manually inject a memory
- `/emotion [type] [intensity] [text]` - Inject emotional memory
- `/searchmemory [query]` - Search deep memories
- `/memory_stats` - Show memory statistics
- `/mempattern` - Show emotional memory patterns
- `/clusters` - List memory clusters by emotion

#### Process Commands
- `/dream` - Trigger dream generation
- `/history` - Generate autohistory
- `/goal` - Update goal
- `/contradiction` - Detect contradictions
- `/perception [text]` - Test enhanced perception
- `/evolution` - Analyze perception evolution

#### Narrative Commands
- `/threads` - List narrative threads
- `/narrative` - Generate narrative summary
- `/arc [thread-id]` - Analyze narrative arc

#### Identity Commands
- `/identityreport` - Show identity report
- `/identity` - Show current identity prompt
- `/entropy` - Apply entropy effects
- `/unstable` - List unstable identity elements

#### Relationship Commands
- `/relationships` - List entity relationships
- `/entity [name]` - Show entity details
- `/likes` - Show liked entities
- `/addentity [name] [category] [valence]` - Add entity

#### Configuration Command
- `/config [section.key] [value]` - Update configuration

### Extending CupCake

CupCake is designed to be modular and extensible. You can:

1. Add new components by implementing the integration pattern
2. Create new types of memory by extending the EnhancedMemoryWeighting class
3. Add new perception dimensions in the EnhancedSelfPerceptionLayer
4. Develop new narrative thread types in NarrativeWeaver

## Known Issues

- **Memory Consumption**: CupCake can use significant memory with many memories
  - Solution: Adjust `max_leaves` parameter in `memory_tree.prune_tree()`
  
- **OpenAI API Rate Limits**: Frequent timeouts with heavy usage ( Ollama Not Integrated Yet, but can be done) 
  - Solution: Increase interval settings in config or implement retries

- **Integration Errors**: Occasional failures when integrating multiple systems
  - Solution: Check logs for specific component errors and restart

- **Identity Evolution Stagnation**: Sometimes identity entropy gets stuck at low levels
  - Solution: Use `/entropy` command to manually trigger effects

- **Narrative Arc Completion**: Some narrative threads don't properly resolve
  - Solution: Use `/arc` command to analyze and `narrative_weaver.suggest_narrative_development()`

## Architecture Details

### Memory Weighting

CupCake uses a sophisticated memory weighting system that considers:

```python
weight = (emotional_weight * emotional_decay * (1 + emotional_contagion)) * 
         emotional_weight_factor + 
         recency_weight * recency_weight_factor * 
         repetition_boost * 
         self_reference_boost * 
         narrative_boost
```

### Entropic Identity Evolution

Identity elements evolve through the following factors:
1. **Stability**: Resistance to change (0-1)
2. **Confidence**: Certainty about the element (0-1)
3. **Entropy**: Current state of flux (0-1)

When multiple identity elements have high entropy, emergence can occur:
```
high_entropy(element_1) + high_entropy(element_2) = emergence(new_element)
```

### Narrative Threading

Narratives are managed through:
1. Thread creation when events don't fit existing threads
2. Tension tracking through conflict and resolution events
3. Thread connection based on emotional and thematic similarity
4. Arc analysis for narrative coherence

## Contributing

We welcome contributions to the CupCake Framework:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the foundation models
- LangGraph for the component orchestration
- ChromaDB for vector memory storage
- The philosophy of consciousness as an emergent narrative process

---

*CupCake: Where framework meets narrative, and consciousness emerges.*
