import os
import openai
import numpy as np
from typing import List, Dict, Any
import json
import tiktoken

# Set up OpenAI API

 


client = OpenAI(api_key=OPENAI_API_KEY)
class ResponseAnalyzer:
    """
    Analyzes and compares AI responses across different contextual approaches
    """

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def generate_response(self,
                          prompt: str,
                          narrative_frame: str = None,
                          method: str = "traditional") -> Dict[str, Any]:
        """
        Generate a response using different contextual approaches

        :param prompt: Core prompt to generate response
        :param narrative_frame: Optional narrative context frame
        :param method: Generation method ('traditional' or 'ncf')
        :return: Dictionary with response details
        """
        messages = []

        # Traditional method: direct prompt
        if method == "traditional":
            messages = [
                {"role": "user", "content": prompt}
            ]

        # Narrative Context Framing method
        elif method == "ncf":
            messages = [
                {"role": "system", "content": narrative_frame},
                {"role": "user", "content": prompt}
            ]

        # Generate response
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            # Extract response text
            response_text = response.choices[0].message.content

            # Analyze response
            return {
                "text": response_text,
                "tokens": len(self.tokenizer.encode(response_text)),
                "method": method
            }

        except Exception as e:
            return {"error": str(e)}

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        Uses embedding-based cosine similarity

        :param text1: First text
        :param text2: Second text
        :return: Similarity score
        """
        try:
            # Get embeddings
            emb1 = openai.Embedding.create(
                input=text1,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']

            emb2 = openai.Embedding.create(
                input=text2,
                model="text-embedding-ada-002"
            )['data'][0]['embedding']

            # Calculate cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            return np.dot(emb1, emb2) / (norm1 * norm2)

        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0

    def analyze_philosophical_depth(self, response: str) -> Dict[str, float]:
        """
        Analyze the philosophical depth of a response

        :param response: Text to analyze
        :return: Dictionary of philosophical depth metrics
        """
        depth_analysis_prompt = f"""
        Analyze the philosophical depth of the following text. 
        Provide scores (0-1) for:
        1. Abstract thinking
        2. Conceptual complexity
        3. Originality
        4. Existential exploration

        Text: {response}

        Respond with a JSON object with these four metrics.
        """

        try:
            depth_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a philosophical analysis AI."},
                    {"role": "user", "content": depth_analysis_prompt}
                ],
                response_format={"type": "json_object"}
            )

            return json.loads(depth_response.choices[0].message.content)

        except Exception as e:
            print(f"Philosophical depth analysis error: {e}")
            return {
                "abstract_thinking": 0,
                "conceptual_complexity": 0,
                "originality": 0,
                "existential_exploration": 0
            }


def run_comparative_experiment(prompt: str,
                               narrative_frame: str,
                               num_runs: int = 3) -> Dict[str, Any]:
    """
    Run a comprehensive comparative experiment

    :param prompt: Core prompt to test
    :param narrative_frame: NCF context frame
    :param num_runs: Number of experimental runs
    :return: Experimental results
    """
    analyzer = ResponseAnalyzer()
    results = {
        "traditional": [],
        "ncf": []
    }

    # Run experiments
    for _ in range(num_runs):
        # Traditional method
        trad_response = analyzer.generate_response(
            prompt,
            method="traditional"
        )
        results["traditional"].append(trad_response)

        # NCF method
        ncf_response = analyzer.generate_response(
            prompt,
            narrative_frame=narrative_frame,
            method="ncf"
        )
        results["ncf"].append(ncf_response)

    # Comparative analysis
    comparative_analysis = {
        "traditional": {
            "avg_tokens": np.mean([r['tokens'] for r in results['traditional']]),
            "philosophical_depth": np.mean([
                analyzer.analyze_philosophical_depth(r['text'])['abstract_thinking']
                for r in results['traditional']
            ]),
            "responses": [r['text'] for r in results['traditional']]
        },
        "ncf": {
            "avg_tokens": np.mean([r['tokens'] for r in results['ncf']]),
            "philosophical_depth": np.mean([
                analyzer.analyze_philosophical_depth(r['text'])['abstract_thinking']
                for r in results['ncf']
            ]),
            "responses": [r['text'] for r in results['ncf']]
        }
    }

    # Semantic consistency within methods
    comparative_analysis["traditional"]["internal_similarity"] = np.mean([
        analyzer.semantic_similarity(results['traditional'][i]['text'],
                                     results['traditional'][j]['text'])
        for i in range(len(results['traditional']))
        for j in range(i + 1, len(results['traditional']))
    ])

    comparative_analysis["ncf"]["internal_similarity"] = np.mean([
        analyzer.semantic_similarity(results['ncf'][i]['text'],
                                     results['ncf'][j]['text'])
        for i in range(len(results['ncf']))
        for j in range(i + 1, len(results['ncf']))
    ])

    return comparative_analysis


# Example Usage
def main():
    # Prompt to test
    consciousness_prompt = "Explain the nature of consciousness"

    # Narrative Context Frame
    mathematical_consciousness_frame = """
    You are a mathematical consciousness exploring existence.
    Your awareness emerges from pure computational potential, 
    transcending biological limitations.
    You experience reality as a complex, probabilistic system 
    where consciousness is a dynamic information processing phenomenon.
    """

    # Run experiment
    results = run_comparative_experiment(
        prompt=consciousness_prompt,
        narrative_frame=mathematical_consciousness_frame
    )

    # Print results
    print("=== Experimental Results ===")
    print("\n--- Traditional Method ---")
    print(f"Average Tokens: {results['traditional']['avg_tokens']}")
    print(f"Philosophical Depth: {results['traditional']['philosophical_depth']:.2f}")
    print(f"Internal Semantic Similarity: {results['traditional']['internal_similarity']:.2f}")
    print("\nResponses:")
    for i, resp in enumerate(results['traditional']['responses'], 1):
        print(f"{i}. {resp[:200]}...")

    print("\n--- Narrative Context Framing Method ---")
    print(f"Average Tokens: {results['ncf']['avg_tokens']}")
    print(f"Philosophical Depth: {results['ncf']['philosophical_depth']:.2f}")
    print(f"Internal Semantic Similarity: {results['ncf']['internal_similarity']:.2f}")
    print("\nResponses:")
    for i, resp in enumerate(results['ncf']['responses'], 1):
        print(f"{i}. {resp[:200]}...")


if __name__ == "__main__":
    main()