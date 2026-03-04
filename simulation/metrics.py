import pandas as pd
import numpy as np

class QualityMetrics:
    """
    DS Role: Statistical evaluation of the Precision-Recall Frontier.
    MLE Role: Benchmarking System Reliability and Economic Scalability.
    """
    def __init__(self, log_df: pd.DataFrame):
        self.df = log_df
        # Industry Standard Benchmarks for the Philippines (2026)
        self.avg_cac = 1500.0  # Cost to acquire one Maya/GCash user (PHP)
        self.churn_prob_per_fp = 0.20  # 20% chance of churn on a False Positive

    def calculate_confusion_matrix(self):
        """Calculates TP, FP, TN, FN for the simulation run."""
        tp = len(self.df[(self.df['is_attack'] == True) & (self.df['blocked'] == True)])
        fp = len(self.df[(self.df['is_attack'] == False) & (self.df['blocked'] == True)])
        tn = len(self.df[(self.df['is_attack'] == False) & (self.df['blocked'] == False)])
        fn = len(self.df[(self.df['is_attack'] == True) & (self.df['blocked'] == False)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {"precision": precision, "recall": recall, "fp_count": fp, "tp_count": tp}

    def calculate_economic_impact(self, avg_fraud_value=25000):
        """
        The 'Gold Standard' Metric: Net Error Revenue (NER).
        NER = (Fraud Prevented) - (Churn Cost)
        """
        stats = self.calculate_confusion_matrix()
        
        total_saved = stats['tp_count'] * avg_fraud_value
        total_churn_cost = (stats['fp_count'] * self.churn_prob_per_fp) * self.avg_cac
        
        ner = total_saved - total_churn_cost
        
        return {
            "net_revenue_saved": ner,
            "roi_ratio": ner / total_churn_cost if total_churn_cost > 0 else float('inf'),
            "friction_ratio": stats['fp_count'] / stats['tp_count'] if stats['tp_count'] > 0 else 0
        }