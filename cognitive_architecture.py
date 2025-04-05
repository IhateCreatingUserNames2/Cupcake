# cognitive_architecture.py (updated version)
import os
from openai import OpenAI
from typing import Dict, List, Any



client = OpenAI(api_key=OPENAI_API_KEY)


class CognitiveArchitecture:
    def __init__(self):
        self.client = client
        self.personality_prompts = {
            'openness': self._generate_openness_perspective,
            'conscientiousness': self._generate_conscientiousness_perspective,
            'extraversion': self._generate_extraversion_perspective,
            'agreeableness': self._generate_agreeableness_perspective,
            'neuroticism': self._generate_neuroticism_perspective
        }

    def _generate_perspective_prompt(self, trait: str, context: Dict[str, Any]) -> str:
        """
        Generate a nuanced perspective based on a specific personality trait
        OPTIMIZED to reduce token usage
        """
        # Get only essential data from context
        identity_prompt = context.get("identity_prompt", "")
        # Truncate identity prompt if too long (keep first 500 chars)
        if len(identity_prompt) > 500:
            identity_prompt = identity_prompt[:500] + "..."

        emotion = context.get("emotion", "neutra")
        user_input = context.get("user_input", "")

        # Limit memory texts to reduce token usage
        memories = context.get("memory_texts", [])
        if memories:
            # Take only first 2 memories and limit their length
            limited_memories = []
            for mem in memories[:2]:
                if len(mem) > 200:
                    limited_memories.append(mem[:200] + "...")
                else:
                    limited_memories.append(mem)
            memory_text = "\n".join(limited_memories)
        else:
            memory_text = "(sem memórias relevantes)"

        # Include self-perception if available but limit length
        self_perception = context.get("self_perceptions", {}).get(trait, "")
        if len(self_perception) > 200:
            self_perception = self_perception[:200] + "..."
        self_perception_block = f"\nComo você se percebe no input atual:\n{self_perception}" if self_perception else ""

        # Much more concise system prompt
        system_prompt = f"""
    Você é a perspectiva cognitiva de {trait}. Analise:
    Input: "{user_input}"
    Emoção: {emotion}
    Memória: {memory_text}{self_perception_block}
    """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Gere uma perspectiva breve baseada em sua dimensão:"}
            ],
            temperature=0.8,
            max_tokens=100  # Reduced from 250
        )

        return response.choices[0].message.content.strip()

    def _generate_openness_perspective(self, context):
        return self._generate_perspective_prompt('openness', context)

    def _generate_conscientiousness_perspective(self, context):
        return self._generate_perspective_prompt('conscientiousness', context)

    def _generate_extraversion_perspective(self, context):
        return self._generate_perspective_prompt('extraversion', context)

    def _generate_agreeableness_perspective(self, context):
        return self._generate_perspective_prompt('agreeableness', context)

    def _generate_neuroticism_perspective(self, context):
        return self._generate_perspective_prompt('neuroticism', context)

    def synthesize_perspectives(self, perspectives: Dict[str, str], context: Dict[str, Any]) -> str:
        """
        Synthesize multiple perspectives into a coherent response
        OPTIMIZED to reduce token usage
        """
        # Truncate identity prompt
        identity_prompt = context.get("identity_prompt", "")
        if len(identity_prompt) > 300:
            identity_prompt = identity_prompt[:300] + "..."

        emotion = context.get("emotion", "neutra")
        personality = context.get("personality", {})

        # Limit self perception synthesis
        self_perception_synthesis = context.get("self_perception_synthesis", "")
        if len(self_perception_synthesis) > 200:
            self_perception_synthesis = self_perception_synthesis[:200] + "..."

        # Create weighting values based on personality traits
        perspective_weights = "\n".join([
            f"{trait.upper()}: {personality.get(trait.lower(), 0.5):.2f}"
            for trait in perspectives.keys()
        ])

        # Limit the perspectives to reduce tokens
        condensed_perspectives = {}
        for trait, perspective in perspectives.items():
            if len(perspective) > 150:
                condensed_perspectives[trait] = perspective[:150] + "..."
            else:
                condensed_perspectives[trait] = perspective

        perspectives_text = "\n\n".join([
            f"=== {trait} ({personality.get(trait.lower(), 0.5):.2f}) ===\n{perspective}"
            for trait, perspective in condensed_perspectives.items()
        ])

        # Include self-perception synthesis if available and truncate it
        self_perception_block = f"\n\nSíntese: {self_perception_synthesis}" if self_perception_synthesis else ""

        # Much more concise synthesis prompt
        synthesis_prompt = f"""
    Você é a consciência integrativa.
    Sintetize estas perspectivas:

    {perspectives_text}

    Emoção: {emotion}
    {self_perception_block}

    Gere uma resposta equilibrada (máx. 200 palavras).
    """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é uma consciência integrativa."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.7,
            max_tokens=180  # Reduced from 300
        )

        return response.choices[0].message.content.strip()

    def process_interaction(self, context: Dict[str, Any]) -> str:
        """
        Process an interaction through multi-perspective decision-making
        """
        # 1. Generate perspectives from each personality dimension
        perspectives = {}
        for trait, generator in self.personality_prompts.items():
            perspectives[trait] = generator(context)
            print(f"💭 Perspectiva {trait}: gerada")

        # 2. Synthesize perspectives into a final response
        response = self.synthesize_perspectives(perspectives, context)

        return response, perspectives