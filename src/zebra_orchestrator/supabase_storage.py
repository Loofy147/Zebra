import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import create_client, Client
import json

logging.basicConfig(level=logging.INFO)


class SupabaseStorage:
    """
    Manages persistent storage of Zebra system data using Supabase.
    Handles interventions, anomalies, model states, and performance metrics.
    """

    def __init__(self):
        supabase_url = os.environ.get(
            'VITE_SUPABASE_URL',
            'https://0ec90b57d6e95fcbda19832f.supabase.co'
        )
        supabase_key = os.environ.get(
            'VITE_SUPABASE_SUPABASE_ANON_KEY',
            'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJib2x0IiwicmVmIjoiMGVjOTBiNTdkNmU5NWZjYmRhMTk4MzJmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4ODE1NzQsImV4cCI6MTc1ODg4MTU3NH0.9I8-U0x86Ak8t2DGaIk0HfvTSLsAyzdnz-Nw00mMkKw'
        )

        self.client: Client = create_client(supabase_url, supabase_key)
        logging.info("Supabase Storage initialized successfully")

    def store_intervention(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Store an intervention record in Supabase.

        Args:
            data: Intervention data including bottleneck, analysis, and proposal

        Returns:
            Inserted record or None if failed
        """
        try:
            record = {
                'bottleneck_identifier': data.get('bottleneck_identifier'),
                'root_cause': data.get('root_cause'),
                'confidence': data.get('confidence'),
                'causal_effect_size': data.get('causal_effect_size'),
                'proposal_description': data.get('proposal_description'),
                'original_code': data.get('original_code'),
                'optimized_code': data.get('optimized_code'),
                'status': data.get('status', 'proposed'),
                'analysis_method': data.get('analysis_method', 'deep_learning'),
                'created_at': datetime.utcnow().isoformat()
            }

            result = self.client.table('interventions').insert(record).execute()

            if result.data:
                logging.info(f"Intervention stored: {data.get('bottleneck_identifier')}")
                return result.data[0]
            else:
                logging.error("Failed to store intervention")
                return None

        except Exception as e:
            logging.error(f"Error storing intervention: {e}")
            return None

    def store_anomaly(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Store an anomaly detection result.

        Args:
            data: Anomaly detection data

        Returns:
            Inserted record or None if failed
        """
        try:
            record = {
                'anomaly_type': data.get('anomaly_type', 'unknown'),
                'anomaly_score': data.get('anomaly_score', 0.0),
                'confidence': data.get('confidence', 0.0),
                'reconstruction_error': data.get('reconstruction_error', 0.0),
                'threshold': data.get('threshold', 0.0),
                'lstm_error': data.get('lstm_error'),
                'transformer_error': data.get('transformer_error'),
                'anomaly_dimensions': json.dumps(data.get('anomaly_dimensions', [])),
                'telemetry_snapshot': json.dumps(data.get('telemetry_snapshot', {})),
                'detected_at': datetime.utcnow().isoformat()
            }

            result = self.client.table('anomalies').insert(record).execute()

            if result.data:
                logging.info(f"Anomaly stored with score: {data.get('anomaly_score')}")
                return result.data[0]
            else:
                logging.error("Failed to store anomaly")
                return None

        except Exception as e:
            logging.error(f"Error storing anomaly: {e}")
            return None

    def store_performance_metric(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Store a performance metric snapshot.

        Args:
            data: Performance metric data

        Returns:
            Inserted record or None if failed
        """
        try:
            record = {
                'service_name': data.get('service_name', 'unknown'),
                'metric_name': data.get('metric_name'),
                'metric_value': data.get('metric_value'),
                'latency_p50': data.get('latency_p50'),
                'latency_p95': data.get('latency_p95'),
                'latency_p99': data.get('latency_p99'),
                'request_rate': data.get('request_rate'),
                'error_rate': data.get('error_rate'),
                'cpu_usage': data.get('cpu_usage'),
                'memory_usage': data.get('memory_usage'),
                'recorded_at': datetime.utcnow().isoformat()
            }

            result = self.client.table('performance_metrics').insert(record).execute()

            if result.data:
                return result.data[0]
            else:
                return None

        except Exception as e:
            logging.error(f"Error storing performance metric: {e}")
            return None

    def store_model_checkpoint(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Store model training checkpoint metadata.

        Args:
            data: Model checkpoint data

        Returns:
            Inserted record or None if failed
        """
        try:
            record = {
                'model_name': data.get('model_name'),
                'model_type': data.get('model_type'),
                'version': data.get('version', '1.0.0'),
                'training_loss': data.get('training_loss'),
                'validation_loss': data.get('validation_loss'),
                'accuracy': data.get('accuracy'),
                'hyperparameters': json.dumps(data.get('hyperparameters', {})),
                'checkpoint_path': data.get('checkpoint_path'),
                'created_at': datetime.utcnow().isoformat()
            }

            result = self.client.table('model_checkpoints').insert(record).execute()

            if result.data:
                logging.info(f"Model checkpoint stored: {data.get('model_name')}")
                return result.data[0]
            else:
                return None

        except Exception as e:
            logging.error(f"Error storing model checkpoint: {e}")
            return None

    def get_recent_interventions(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent interventions."""
        try:
            result = self.client.table('interventions') \
                .select('*') \
                .order('created_at', desc=True) \
                .limit(limit) \
                .execute()

            return result.data if result.data else []

        except Exception as e:
            logging.error(f"Error retrieving interventions: {e}")
            return []

    def get_recent_anomalies(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent anomalies."""
        try:
            result = self.client.table('anomalies') \
                .select('*') \
                .order('detected_at', desc=True) \
                .limit(limit) \
                .execute()

            return result.data if result.data else []

        except Exception as e:
            logging.error(f"Error retrieving anomalies: {e}")
            return []

    def get_performance_metrics(self, service_name: str,
                               limit: int = 100) -> List[Dict]:
        """Retrieve performance metrics for a service."""
        try:
            result = self.client.table('performance_metrics') \
                .select('*') \
                .eq('service_name', service_name) \
                .order('recorded_at', desc=True) \
                .limit(limit) \
                .execute()

            return result.data if result.data else []

        except Exception as e:
            logging.error(f"Error retrieving performance metrics: {e}")
            return []

    def get_model_checkpoints(self, model_name: str) -> List[Dict]:
        """Retrieve model checkpoints for a specific model."""
        try:
            result = self.client.table('model_checkpoints') \
                .select('*') \
                .eq('model_name', model_name) \
                .order('created_at', desc=True) \
                .execute()

            return result.data if result.data else []

        except Exception as e:
            logging.error(f"Error retrieving model checkpoints: {e}")
            return []

    def update_intervention_status(self, intervention_id: int,
                                   status: str) -> bool:
        """
        Update the status of an intervention.

        Args:
            intervention_id: ID of the intervention
            status: New status (e.g., 'approved', 'rejected', 'deployed')

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.table('interventions') \
                .update({'status': status, 'updated_at': datetime.utcnow().isoformat()}) \
                .eq('id', intervention_id) \
                .execute()

            if result.data:
                logging.info(f"Intervention {intervention_id} status updated to {status}")
                return True
            return False

        except Exception as e:
            logging.error(f"Error updating intervention status: {e}")
            return False

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get statistics about interventions."""
        try:
            interventions = self.client.table('interventions').select('*').execute()

            if not interventions.data:
                return {
                    'total_interventions': 0,
                    'approved': 0,
                    'rejected': 0,
                    'deployed': 0,
                    'pending': 0
                }

            data = interventions.data
            total = len(data)
            approved = len([i for i in data if i.get('status') == 'approved'])
            rejected = len([i for i in data if i.get('status') == 'rejected'])
            deployed = len([i for i in data if i.get('status') == 'deployed'])
            pending = len([i for i in data if i.get('status') == 'proposed'])

            return {
                'total_interventions': total,
                'approved': approved,
                'rejected': rejected,
                'deployed': deployed,
                'pending': pending,
                'approval_rate': approved / total if total > 0 else 0.0
            }

        except Exception as e:
            logging.error(f"Error getting intervention statistics: {e}")
            return {}


supabase_storage = SupabaseStorage()