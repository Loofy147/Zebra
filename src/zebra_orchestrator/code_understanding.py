import logging
import torch
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)


class CodeBERTAnalyzer:
    """
    Uses CodeBERT/GraphCodeBERT for code understanding and analysis.
    Extracts semantic embeddings and analyzes code patterns.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logging.info(f"CodeBERT model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load CodeBERT model: {e}")
            self.tokenizer = None
            self.model = None

    def encode_code(self, code: str) -> Optional[np.ndarray]:
        """
        Generate semantic embedding for code snippet.

        Args:
            code: Source code string

        Returns:
            Embedding vector or None if failed
        """
        if not self.model or not self.tokenizer:
            return None

        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embeddings[0]

        except Exception as e:
            logging.error(f"Error encoding code: {e}")
            return None

    def compute_similarity(self, code1: str, code2: str) -> float:
        """
        Compute semantic similarity between two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.encode_code(code1)
        emb2 = self.encode_code(code2)

        if emb1 is None or emb2 is None:
            return 0.0

        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def find_similar_patterns(self, target_code: str,
                            code_corpus: List[str]) -> List[Tuple[int, float]]:
        """
        Find similar code patterns in a corpus.

        Args:
            target_code: Code to search for
            code_corpus: List of code snippets to search in

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        target_emb = self.encode_code(target_code)
        if target_emb is None:
            return []

        similarities = []
        for idx, code in enumerate(code_corpus):
            emb = self.encode_code(code)
            if emb is not None:
                sim = cosine_similarity([target_emb], [emb])[0][0]
                similarities.append((idx, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def analyze_code_quality(self, code: str) -> Dict[str, any]:
        """
        Analyze code quality using semantic understanding.

        Returns:
            Quality metrics and recommendations
        """
        embedding = self.encode_code(code)

        if embedding is None:
            return {'error': 'Failed to analyze code'}

        complexity_score = np.linalg.norm(embedding) / 100.0

        has_error_handling = 'try' in code or 'except' in code or 'catch' in code
        has_logging = 'logging' in code or 'logger' in code or 'log.' in code
        has_comments = '#' in code or '"""' in code or '//' in code

        quality_score = 0.0
        if has_error_handling:
            quality_score += 0.3
        if has_logging:
            quality_score += 0.2
        if has_comments:
            quality_score += 0.2
        if complexity_score < 5.0:
            quality_score += 0.3

        return {
            'quality_score': min(quality_score, 1.0),
            'complexity': float(complexity_score),
            'has_error_handling': has_error_handling,
            'has_logging': has_logging,
            'has_documentation': has_comments,
            'embedding_norm': float(np.linalg.norm(embedding))
        }


class CodeT5Generator:
    """
    Uses CodeT5 for code generation, optimization, and transformation tasks.
    Capable of generating code from natural language and refactoring existing code.
    """

    def __init__(self, model_name: str = "Salesforce/codet5-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logging.info(f"CodeT5 model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load CodeT5 model: {e}")
            self.tokenizer = None
            self.model = None

    def generate_code(self, prompt: str, max_length: int = 256) -> Optional[str]:
        """
        Generate code from natural language prompt.

        Args:
            prompt: Natural language description
            max_length: Maximum length of generated code

        Returns:
            Generated code or None if failed
        """
        if not self.model or not self.tokenizer:
            return None

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True
                )

            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_code

        except Exception as e:
            logging.error(f"Error generating code: {e}")
            return None

    def refactor_code(self, code: str, instruction: str) -> Optional[str]:
        """
        Refactor code based on instruction.

        Args:
            code: Original code
            instruction: Refactoring instruction

        Returns:
            Refactored code or None if failed
        """
        prompt = f"{instruction}: {code}"
        return self.generate_code(prompt)

    def optimize_for_performance(self, code: str) -> Optional[str]:
        """
        Optimize code for better performance.

        Args:
            code: Original code

        Returns:
            Optimized code or None if failed
        """
        prompt = f"Optimize this code for performance: {code}"
        return self.generate_code(prompt)

    def add_error_handling(self, code: str) -> Optional[str]:
        """
        Add error handling to code.

        Args:
            code: Original code

        Returns:
            Code with error handling or None if failed
        """
        prompt = f"Add comprehensive error handling to this code: {code}"
        return self.generate_code(prompt)

    def document_code(self, code: str) -> Optional[str]:
        """
        Generate documentation for code.

        Args:
            code: Code to document

        Returns:
            Documented code or None if failed
        """
        prompt = f"Add detailed docstrings and comments to this code: {code}"
        return self.generate_code(prompt)


class AdvancedCodeAnalyzer:
    """
    Integrates CodeBERT and CodeT5 for comprehensive code analysis and generation.
    Provides high-level API for the Zebra system.
    """

    def __init__(self):
        self.codebert = CodeBERTAnalyzer()
        self.codet5 = CodeT5Generator()
        logging.info("Advanced Code Analyzer initialized")

    def analyze_bottleneck(self, code: str, bottleneck_description: str) -> Dict[str, any]:
        """
        Perform comprehensive analysis of a code bottleneck.

        Args:
            code: Code containing the bottleneck
            bottleneck_description: Description of the performance issue

        Returns:
            Analysis results with quality metrics and recommendations
        """
        quality_analysis = self.codebert.analyze_code_quality(code)

        embedding = self.codebert.encode_code(code)

        analysis = {
            'original_code': code,
            'bottleneck_description': bottleneck_description,
            'quality_metrics': quality_analysis,
            'has_embedding': embedding is not None,
            'recommendations': []
        }

        if quality_analysis.get('complexity', 0) > 7.0:
            analysis['recommendations'].append('High complexity detected - consider refactoring')

        if not quality_analysis.get('has_error_handling', False):
            analysis['recommendations'].append('Add error handling for robustness')

        if not quality_analysis.get('has_logging', False):
            analysis['recommendations'].append('Add logging for observability')

        return analysis

    def generate_optimized_version(self, code: str,
                                  optimization_goal: str) -> Dict[str, any]:
        """
        Generate optimized version of code.

        Args:
            code: Original code
            optimization_goal: Goal of optimization (e.g., 'performance', 'readability')

        Returns:
            Optimization results
        """
        if optimization_goal == 'performance':
            optimized = self.codet5.optimize_for_performance(code)
        elif optimization_goal == 'error_handling':
            optimized = self.codet5.add_error_handling(code)
        elif optimization_goal == 'documentation':
            optimized = self.codet5.document_code(code)
        else:
            instruction = f"Refactor for {optimization_goal}"
            optimized = self.codet5.refactor_code(code, instruction)

        result = {
            'original_code': code,
            'optimized_code': optimized,
            'optimization_goal': optimization_goal,
            'success': optimized is not None
        }

        if optimized:
            similarity = self.codebert.compute_similarity(code, optimized)
            result['semantic_similarity'] = similarity
            result['significant_change'] = similarity < 0.9

        return result

    def compare_implementations(self, code1: str, code2: str) -> Dict[str, any]:
        """
        Compare two code implementations.

        Args:
            code1: First implementation
            code2: Second implementation

        Returns:
            Comparison results
        """
        similarity = self.codebert.compute_similarity(code1, code2)
        quality1 = self.codebert.analyze_code_quality(code1)
        quality2 = self.codebert.analyze_code_quality(code2)

        return {
            'semantic_similarity': similarity,
            'code1_quality': quality1.get('quality_score', 0),
            'code2_quality': quality2.get('quality_score', 0),
            'code1_complexity': quality1.get('complexity', 0),
            'code2_complexity': quality2.get('complexity', 0),
            'recommended': 1 if quality1.get('quality_score', 0) > quality2.get('quality_score', 0) else 2
        }

    def extract_code_features(self, code: str) -> Dict[str, any]:
        """
        Extract comprehensive features from code for ML models.

        Args:
            code: Source code

        Returns:
            Feature dictionary
        """
        embedding = self.codebert.encode_code(code)
        quality = self.codebert.analyze_code_quality(code)

        features = {
            'embedding': embedding.tolist() if embedding is not None else None,
            'embedding_dim': len(embedding) if embedding is not None else 0,
            'quality_score': quality.get('quality_score', 0),
            'complexity': quality.get('complexity', 0),
            'has_error_handling': quality.get('has_error_handling', False),
            'has_logging': quality.get('has_logging', False),
            'has_documentation': quality.get('has_documentation', False),
            'line_count': len(code.split('\n')),
            'char_count': len(code)
        }

        return features


code_analyzer = AdvancedCodeAnalyzer()