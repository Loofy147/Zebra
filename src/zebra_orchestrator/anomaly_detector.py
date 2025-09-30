import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for time-series telemetry data.
    Uses autoencoder architecture to learn normal behavior patterns.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMAnomalyDetector, self).__init__()

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]

        Returns:
            Tuple of (reconstructed_sequence, latent_representation)
        """
        encoded, (hidden, cell) = self.encoder(x)

        decoded, _ = self.decoder(encoded, (hidden, cell))

        reconstructed = self.output_layer(decoded)

        return reconstructed, encoded


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector for complex temporal patterns.
    More powerful than LSTM for long-range dependencies.
    """

    def __init__(self, input_dim: int = 7, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super(TransformerAnomalyDetector, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]

        Returns:
            Reconstructed sequence
        """
        x = self.input_projection(x)

        encoded = self.transformer_encoder(x)

        reconstructed = self.output_projection(encoded)

        return reconstructed


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple models for robust detection.
    Uses LSTM and Transformer models with voting mechanism.
    """

    def __init__(self, window_size: int = 10, threshold_percentile: float = 95.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile

        self.lstm_model = LSTMAnomalyDetector().to(self.device)
        self.transformer_model = TransformerAnomalyDetector().to(self.device)

        self.scaler = StandardScaler()
        self.is_fitted = False

        self.data_buffer = deque(maxlen=window_size)
        self.reconstruction_errors = []

        self.lstm_optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.transformer_optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001)

        logging.info(f"Ensemble Anomaly Detector initialized on {self.device}")

    def extract_features(self, telemetry_data: Dict) -> np.ndarray:
        """Extract numerical features from telemetry."""
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

    def update_buffer(self, features: np.ndarray):
        """Add new data point to the sliding window buffer."""
        self.data_buffer.append(features)

    def fit_scaler(self, historical_data: List[np.ndarray]):
        """Fit the scaler on historical normal data."""
        if len(historical_data) > 0:
            data_matrix = np.vstack(historical_data)
            self.scaler.fit(data_matrix)
            self.is_fitted = True
            logging.info("Scaler fitted on historical data")

    def train_step(self, sequence: torch.Tensor) -> Dict[str, float]:
        """
        Train both models on a sequence of normal data.

        Args:
            sequence: Input sequence [batch_size, seq_len, input_dim]

        Returns:
            Dictionary of training losses
        """
        self.lstm_model.train()
        self.transformer_model.train()

        self.lstm_optimizer.zero_grad()
        lstm_reconstructed, _ = self.lstm_model(sequence)
        lstm_loss = nn.MSELoss()(lstm_reconstructed, sequence)
        lstm_loss.backward()
        self.lstm_optimizer.step()

        self.transformer_optimizer.zero_grad()
        transformer_reconstructed = self.transformer_model(sequence)
        transformer_loss = nn.MSELoss()(transformer_reconstructed, sequence)
        transformer_loss.backward()
        self.transformer_optimizer.step()

        return {
            'lstm_loss': lstm_loss.item(),
            'transformer_loss': transformer_loss.item(),
            'combined_loss': (lstm_loss.item() + transformer_loss.item()) / 2
        }

    def detect_anomaly(self, telemetry_data: Dict) -> Dict[str, any]:
        """
        Detect if current telemetry represents an anomaly.

        Args:
            telemetry_data: Current telemetry data

        Returns:
            Detection result with anomaly score and classification
        """
        features = self.extract_features(telemetry_data)
        self.update_buffer(features)

        if len(self.data_buffer) < self.window_size:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'reason': 'insufficient_data',
                'timestamp': datetime.utcnow().isoformat()
            }

        sequence = np.array(list(self.data_buffer))

        if self.is_fitted:
            sequence = self.scaler.transform(sequence)

        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        self.lstm_model.eval()
        self.transformer_model.eval()

        with torch.no_grad():
            lstm_reconstructed, lstm_latent = self.lstm_model(sequence_tensor)
            lstm_error = torch.mean((lstm_reconstructed - sequence_tensor) ** 2).item()

            transformer_reconstructed = self.transformer_model(sequence_tensor)
            transformer_error = torch.mean((transformer_reconstructed - sequence_tensor) ** 2).item()

            ensemble_error = (lstm_error + transformer_error) / 2

        self.reconstruction_errors.append(ensemble_error)

        if len(self.reconstruction_errors) > 100:
            self.reconstruction_errors = self.reconstruction_errors[-100:]

        if len(self.reconstruction_errors) >= 10:
            threshold = np.percentile(self.reconstruction_errors, self.threshold_percentile)
            is_anomaly = ensemble_error > threshold
            normalized_score = min(ensemble_error / (threshold + 1e-6), 10.0)
        else:
            is_anomaly = ensemble_error > 0.5
            threshold = 0.5
            normalized_score = ensemble_error

        confidence = min(ensemble_error / (threshold + 1e-6), 1.0) if is_anomaly else \
                     1.0 - min(ensemble_error / (threshold + 1e-6), 1.0)

        anomaly_details = self._analyze_anomaly_dimensions(sequence_tensor, lstm_reconstructed)

        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(normalized_score),
            'confidence': float(confidence),
            'reconstruction_error': float(ensemble_error),
            'threshold': float(threshold),
            'lstm_error': float(lstm_error),
            'transformer_error': float(transformer_error),
            'anomaly_dimensions': anomaly_details,
            'timestamp': datetime.utcnow().isoformat()
        }

        if is_anomaly:
            logging.warning(
                f"ANOMALY DETECTED | Score: {normalized_score:.3f} | "
                f"Error: {ensemble_error:.4f} | Threshold: {threshold:.4f}"
            )

        return result

    def _analyze_anomaly_dimensions(self, original: torch.Tensor,
                                   reconstructed: torch.Tensor) -> List[Dict]:
        """
        Analyze which dimensions contribute most to the anomaly.

        Returns:
            List of dimension analyses
        """
        dimension_names = [
            'latency_p50', 'latency_p95', 'latency_p99',
            'request_rate', 'error_rate', 'cpu_usage', 'memory_usage'
        ]

        dimension_errors = torch.mean((original - reconstructed) ** 2, dim=1).squeeze()

        analyses = []
        for idx, name in enumerate(dimension_names):
            if idx < len(dimension_errors):
                error = dimension_errors[idx].item()
                analyses.append({
                    'dimension': name,
                    'error': float(error),
                    'is_primary_contributor': error > dimension_errors.mean().item()
                })

        analyses.sort(key=lambda x: x['error'], reverse=True)
        return analyses

    def save_models(self, path_prefix: str):
        """Save both model checkpoints."""
        torch.save({
            'lstm_state': self.lstm_model.state_dict(),
            'transformer_state': self.transformer_model.state_dict(),
            'lstm_optimizer': self.lstm_optimizer.state_dict(),
            'transformer_optimizer': self.transformer_optimizer.state_dict(),
            'scaler': self.scaler,
            'reconstruction_errors': self.reconstruction_errors
        }, f"{path_prefix}_ensemble.pt")
        logging.info(f"Models saved to {path_prefix}_ensemble.pt")

    def load_models(self, path_prefix: str):
        """Load model checkpoints."""
        checkpoint = torch.load(f"{path_prefix}_ensemble.pt", map_location=self.device)
        self.lstm_model.load_state_dict(checkpoint['lstm_state'])
        self.transformer_model.load_state_dict(checkpoint['transformer_state'])
        self.lstm_optimizer.load_state_dict(checkpoint['lstm_optimizer'])
        self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer'])
        self.scaler = checkpoint['scaler']
        self.reconstruction_errors = checkpoint['reconstruction_errors']
        self.is_fitted = True
        logging.info(f"Models loaded from {path_prefix}_ensemble.pt")


anomaly_detector = EnsembleAnomalyDetector()