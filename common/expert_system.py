class RuleBasedExpertSystem:
    def __init__(self):
        self.rules = {
            'high_amount': 10000,
            'high_balance_change': 5000,
            'suspicious_ratio': 0.8,
            'multiple_transactions': 3,
            'time_threshold': 3600  # 1 hour in seconds
        }
    
    def apply_rules(self, X, y=None):
        """
        Apply domain-specific rules to score transactions for fraud likelihood.
        Returns a score between 0 and 1 for each transaction.
        """
        expert_scores = []
        
        for idx, row in X.iterrows():
            score = 0.0
            reasons = []
            
            # Rule 1: High transaction amount
            if row['amount'] > self.rules['high_amount']:
                score += 0.3
                reasons.append('high_amount')
            
            # Rule 2: Large balance changes
            balance_change_orig = abs(row['oldbalanceOrg'] - row['newbalanceOrig'])
            balance_change_dest = abs(row['oldbalanceDest'] - row['newbalanceDest'])
            
            if balance_change_orig > self.rules['high_balance_change'] or \
               balance_change_dest > self.rules['high_balance_change']:
                score += 0.2
                reasons.append('high_balance_change')
            
            # Rule 3: Suspicious transaction ratio
            if row['transaction_ratio'] > self.rules['suspicious_ratio']:
                score += 0.2
                reasons.append('suspicious_ratio')
            
            # Rule 4: Type-specific rules
            if row['type'] in [1, 3]:  # CASH_OUT or TRANSFER
                if row['amount'] > self.rules['high_amount'] * 0.5:
                    score += 0.15
                    reasons.append('suspicious_type')
            
            # Rule 5: Balance depletion
            if row['newbalanceOrig'] < row['amount'] * 0.1:
                score += 0.15
                reasons.append('balance_depletion')
            
            # Cap the score at 1.0
            score = min(score, 1.0)
            expert_scores.append({
                'score': score,
                'reasons': reasons
            })
        
        return expert_scores
    
    def get_rule_explanations(self):
        """
        Returns explanations for all rules used in the system.
        """
        return {
            'high_amount': f"Transactions above {self.rules['high_amount']} are considered high-risk",
            'high_balance_change': f"Balance changes above {self.rules['high_balance_change']} are suspicious",
            'suspicious_ratio': f"Transaction amount ratio above {self.rules['suspicious_ratio']} indicates potential fraud",
            'suspicious_type': "Certain transaction types (CASH_OUT, TRANSFER) are inherently more risky",
            'balance_depletion': "Transactions that nearly deplete the account balance are suspicious"
        }
