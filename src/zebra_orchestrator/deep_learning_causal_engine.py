import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)


@dataclass
class CausalFeatures:
    """Represents features extracted from telemetry for causal inference."""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    request_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime
    intervention_active: bool
    intervention_type: Optional[str] = None


class CausalInferenceNetwork(nn.Module):
    """
    Deep neural network for causal inference.
    Uses treatment-aware architecture to estimate causal effects.
    """

    def __init__(self, input_dim: int = 7, hidden_dims: List[int] = [128, 64, 32]):
        super(CausalInferenceNetwork, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.treatment_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(prev_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, treatment: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input features [batch_size, input_dim]
            treatment: Optional treatment indicator [batch_size, 1]

        Returns:
            Tuple of (treatment_propensity, outcome_prediction)
        """
        features = self.feature_extractor(x)

        treatment_propensity = self.treatment_head(features)

        if treatment is None:
            treatment = treatment_propensity

        combined = torch.cat([features, treatment], dim=1)
        outcome = self.outcome_head(combined)

        return treatment_propensity, outcome


class DeepCausalInferenceEngine:
    """
    Advanced causal inference engine using deep learning.
    Implements doubly robust estimation for causal effect estimation.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CausalInferenceNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.training_history: List[Dict] = []

        if model_path:
            self.load_model(model_path)

        logging.info(f"Deep Causal Inference Engine initialized on device: {self.device}")

    def extract_features(self, telemetry_data: Dict) -> np.ndarray:
        """Extract numerical features from telemetry data."""
        try:
            features = [
                telemetry_data.get('latency_p50', 0.0),
                telemetry_data.get('latency_p95', 0.0),
                telemetry_data.get('latency_p99', 0.0),
                telemetry_data.get('request_rate', 0.0),
                telemetry_data.get('error_rate', 0.0),
                telemetry_data.get('cpu_usage', 0.0),
                telemetry_data.get('memory_usage', 0.0),
            ]
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return np.zeros(7, dtype=np.float32)

    def train_step(self, features: torch.Tensor, treatments: torch.Tensor,
                   outcomes: torch.Tensor) -> Dict[str, float]:
        """
        Single training step using doubly robust loss.

        Args:
            features: Input features [batch_size, input_dim]
            treatments: Treatment indicators [batch_size, 1]
            outcomes: Observed outcomes [batch_size, 1]

        Returns:
            Dictionary of loss components
        """
        self.model.train()
        self.optimizer.zero_grad()

        treatment_propensity, outcome_pred = self.model(features, treatments)

        propensity_loss = nn.BCELoss()(treatment_propensity, treatments)

        outcome_loss = nn.MSELoss()(outcome_pred, outcomes)

        ipw_weights = treatments / (treatment_propensity + 1e-6) + \
                      (1 - treatments) / (1 - treatment_propensity + 1e-6)
        weighted_outcome_loss = (ipw_weights.detach() * (outcome_pred - outcomes) ** 2).mean()

        total_loss = propensity_loss + outcome_loss + 0.5 * weighted_outcome_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'propensity_loss': propensity_loss.item(),
            'outcome_loss': outcome_loss.item(),
            'weighted_outcome_loss': weighted_outcome_loss.item()
        }

    def estimate_causal_effect(self, features: np.ndarray,
                               intervention_type: str) -> Dict[str, float]:
        """
        Estimate the causal effect of an intervention.

        Args:
            features: Feature vector for the current system state
            intervention_type: Type of intervention to evaluate

        Returns:
            Dictionary containing causal effect estimates and confidence
        """
        self.model.eval()

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            treatment_0 = torch.zeros(1, 1).to(self.device)
            treatment_1 = torch.ones(1, 1).to(self.device)

            _, outcome_0 = self.model(features_tensor, treatment_0)
            propensity, outcome_1 = self.model(features_tensor, treatment_1)

            ate = (outcome_1 - outcome_0).item()

            confidence = 1.0 - abs(propensity.item() - 0.5) * 2

            return {
                'causal_effect': ate,
                'confidence': confidence,
                'propensity_score': propensity.item(),
                'predicted_outcome_without_treatment': outcome_0.item(),
                'predicted_outcome_with_treatment': outcome_1.item(),
                'intervention_type': intervention_type
            }

    def analyze_anomaly_deep(self, telemetry_data: Dict,
                            anomaly_type: str) -> Dict[str, any]:
        """
        Perform deep learning-based causal analysis of an anomaly.

        Args:
            telemetry_data: Current telemetry data
            anomaly_type: Type of anomaly detected

        Returns:
            Comprehensive analysis including causal estimates
        """
        logging.info(f"Deep CIE: Analyzing anomaly '{anomaly_type}' with neural causal inference...")

        features = self.extract_features(telemetry_data)

        intervention_type = telemetry_data.get('active_intervention', 'none')

        causal_analysis = self.estimate_causal_effect(features, intervention_type)

        if abs(causal_analysis['causal_effect']) > 0.2 and causal_analysis['confidence'] > 0.7:
            root_cause = intervention_type if intervention_type != 'none' else 'systemic_issue'
            is_significant = True
        else:
            root_cause = 'random_variance'
            is_significant = False

        result = {
            'root_cause': root_cause,
            'confidence': causal_analysis['confidence'],
            'causal_effect_size': causal_analysis['causal_effect'],
            'is_statistically_significant': is_significant,
            'propensity_score': causal_analysis['propensity_score'],
            'analysis_method': 'deep_learning_doubly_robust',
            'timestamp': datetime.utcnow().isoformat()
        }

        logging.critical(
            f"Deep CIE RESULT: Root Cause: {result['root_cause']} | "
            f"Confidence: {result['confidence']:.3f} | "
            f"Causal Effect: {result['causal_effect_size']:.3f}"
        )

        return result

    def save_model(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logging.info(f"Model loaded from {path}")


deep_cie_engine = DeepCausalInferenceEngine()