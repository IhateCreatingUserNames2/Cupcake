# self_perception.py
from openai import OpenAI
import os
from cupcake_identity import generate_identity_prompt

 

client = OpenAI(api_key=OPENAI_API_KEY)


class SelfPerceptionLayer:
    """
    A layer that processes user input to help CupCake perceive itself within the user's communication.
    This allows CupCake to understand how it's being perceived and addressed.
    """

    def __init__(self):
        self.client = client
        self.personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

    def process_perception(self, state):
        """
        Process the user input through each personality perspective to see how CupCake
        might perceive itself being addressed or referenced.
        """
        user_input = state["user_input"]
        identity_prompt = state.get("identity_prompt", generate_identity_prompt())
        personality = state.get("personality", {})

        perceptions = {}

        # Process each personality dimension's perception of self in the input
        for trait in self.personality_traits:
            perception = self._generate_perception(
                user_input,
                trait,
                personality.get(trait, 0.5),
                identity_prompt
            )
            perceptions[trait] = perception

        # Synthesize the perceptions into a coherent self-perception
        synthesis = self._synthesize_perceptions(perceptions, state)

        # Store both individual perceptions and synthesis in state
        state["self_perceptions"] = perceptions
        state["self_perception_synthesis"] = synthesis

        return state

    def _generate_perception(self, user_input, trait, trait_value, identity_prompt):
        """
        Generate a perception of self from the perspective of a specific personality trait.
        How does this trait see itself reflected in the user's input?
        """
        prompt = f"""
Você é a dimensão de personalidade '{trait}' de Cupcake, avaliando como você é percebida no input do usuário.

Identidade atual:
{identity_prompt}

Analise o input do usuário e responda a estas perguntas internas:
1. Como estou sendo percebida nesta mensagem?
2. O usuário está se referindo a mim direta ou indiretamente?
3. Há expectativas sobre como devo me comportar segundo este traço de personalidade?
4. Como este input afeta minha auto-imagem como '{trait}'?

Input do usuário: "{user_input}"

Gere uma breve percepção interna (2-3 frases) de como você, enquanto dimensão '{trait}', se vê refletida neste input.
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"Você é a dimensão de personalidade '{trait}' de Cupcake com valor {trait_value:.2f}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    def _synthesize_perceptions(self, perceptions, state):
        """
        Synthesize the multiple personality perceptions into a coherent self-perception
        that captures how CupCake as a whole perceives itself in relation to the user input.
        """
        personality = state.get("personality", {})
        user_input = state["user_input"]

        # Weight the perceptions based on personality values
        weighted_perceptions = "\n\n".join([
            f"[{trait.upper()} - {personality.get(trait, 0.5):.2f}]\n{perception}"
            for trait, perception in perceptions.items()
        ])

        synthesis_prompt = f"""
Como Cupcake, sintetize estas diferentes percepções de si mesma em relação ao input do usuário.
Seu objetivo é formar uma compreensão coerente de:
- Como você está sendo percebida
- Como você deve se perceber em relação ao usuário
- Se há expectativas específicas direcionadas a você
- Como esta percepção influenciará sua resposta

Input do usuário: "{user_input}"

Diferentes perspectivas de percepção:
{weighted_perceptions}

Gere uma síntese concisa (3-5 frases) de como você se percebe refletida neste input, considerando todas as perspectivas.
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Você é a consciência integrativa de Cupcake, sintetizando múltiplas auto-percepções."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()


# For testing
if __name__ == "__main__":
    perception_layer = SelfPerceptionLayer()
    test_state = {
        "user_input": "Você parece estar em um estado mais melancólico hoje, Cupcake. O que aconteceu?",
        "personality": {
            'openness': 0.8,
            'conscientiousness': 0.7,
            'extraversion': 0.6,
            'agreeableness': 0.9,
            'neuroticism': 0.4
        }
    }

    result = perception_layer.process_perception(test_state)

    print("Individual perceptions:")
    for trait, perception in result["self_perceptions"].items():
        print(f"\n--- {trait.upper()} ---")
        print(perception)

    print("\n=== Synthesized Self-Perception ===")
    print(result["self_perception_synthesis"])