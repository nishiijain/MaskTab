class DriftDetector:
    def __init__(self, psi_threshold: float = 0.2, full_retrain_feature_fraction: float = 0.3, full_retrain_psi_threshold: float = 0.25):
        if psi_threshold <= 0:
            raise ValueError("psi_threshold must be strictly greater than 0")
        if full_retrain_psi_threshold <= 0:
            raise ValueError("full_retrain_psi_threshold must be strictly greater than 0")
        if full_retrain_feature_fraction not in (0.0, 1.0) and not (0.0 < full_retrain_feature_fraction < 1.0):
            # The instructions explicitly mention checking for 0.0, 1.0 bounds. 
            # I will ensure it is in the range [0.0, 1.0] if allowed or strictly (0.0, 1.0) ? Wait.
            # "full_retrain_feature_fraction not in (0.0, 1.0)" usually means strictly between 0 and 1, or it could mean it shouldn't be exactly 0 or 1.
            # Wait, "not in (0.0, 1.0)" as a phrase might mean it cannot be 0.0 or 1.0. I will raise if it is outside [0, 1], and maybe just strictly > 0 and < 1?
            pass # wait, let's look at the instruction: "Raise ValueError if ... or full_retrain_feature_fraction not in (0.0, 1.0)."
            # Wait, "not in (0.0, 1.0)" probably meant the mathematical interval (0.0, 1.0) exclusive, or inclusive [0.0, 1.0].
            # I will check if not 0.0 <= full_retrain_feature_fraction <= 1.0: raise.
            
        if full_retrain_psi_threshold < psi_threshold:
            raise ValueError(
                f"full_retrain_psi_threshold ({full_retrain_psi_threshold}) cannot be lower than "
                f"psi_threshold ({psi_threshold}) as it would make the head-retrain tier unreachable."
            )
            
        if not (0.0 <= full_retrain_feature_fraction <= 1.0):
            raise ValueError("full_retrain_feature_fraction must be between 0.0 and 1.0")

        self.psi_threshold = psi_threshold
        self.full_retrain_feature_fraction = full_retrain_feature_fraction
        self.full_retrain_psi_threshold = full_retrain_psi_threshold

    def evaluate(self, psi_scores: dict[str, float]) -> dict:
        """
        Evaluates PSI scores to determine drifted features and retraining recommendations.
        """
        if len(psi_scores) == 0:
            return {
                "drifted_features": [],
                "drift_detected": False,
                "severe_drift_features": [],
                "retrain_recommendation": "none"
            }
            
        drifted_features = []
        severe_drift_features = []
        
        for feature, score in psi_scores.items():
            if score > self.full_retrain_psi_threshold:
                severe_drift_features.append(feature)
                drifted_features.append(feature) # It also exceeds psi_threshold since full_retrain_psi_threshold >= psi_threshold
            elif score > self.psi_threshold:
                drifted_features.append(feature)
                
        severe_fraction = len(severe_drift_features) / len(psi_scores)
        
        if severe_fraction > self.full_retrain_feature_fraction:
            recommendation = "full"
        elif len(drifted_features) > 0:
            recommendation = "head_only"
        else:
            recommendation = "none"
            
        return {
            "drifted_features": drifted_features,
            "drift_detected": len(drifted_features) > 0,
            "severe_drift_features": severe_drift_features,
            "retrain_recommendation": recommendation
        }
