import logging

class RuleBasedExpertSystem:
    def __init__(self):
        self.rules = [
            {
                'name': 'High Amount Rule',
                'condition': lambda x: x['amount'] > 10000,
                'weight': 0.8,
                'description': 'Transaction amount exceeds threshold'
            },
            {
                'name': 'New Merchant Rule',
                'condition': lambda x: x['merchant'] not in ['M1', 'M2', 'M3'],
                'weight': 0.6,
                'description': 'Transaction with new merchant'
            },
            {
                'name': 'Time Rule',
                'condition': lambda x: x['hour'] < 6 or x['hour'] > 22,
                'weight': 0.7,
                'description': 'Transaction during unusual hours'
            }
        ]
        self.performance_history = []
    
    def adjust_rule_weights(self, model_performance):
        """Dynamically adjust rule weights based on model performance"""
        try:
            self.performance_history.append(model_performance)
            
            # Calculate performance trend
            if len(self.performance_history) >= 2:
                trend = self.performance_history[-1]['auc'] - self.performance_history[-2]['auc']
                
                # Adjust weights based on trend
                for rule in self.rules:
                    if trend < 0:  # Performance decreasing
                        rule['weight'] *= 1.1  # Increase rule influence
                    else:  # Performance improving
                        rule['weight'] *= 0.9  # Decrease rule influence
                    
                    # Keep weights in reasonable range
                    rule['weight'] = max(0.1, min(1.0, rule['weight']))
                    
                logging.info("Adjusted expert rule weights based on performance trend")
                
        except Exception as e:
            logging.error(f"Error adjusting rule weights: {str(e)}")
    
    def apply_rules(self, data):
        """Apply expert rules to data"""
        try:
            results = []
            for _, row in data.iterrows():
                score = 0
                reasons = []
                total_weight = 0
                
                for rule in self.rules:
                    if rule['condition'](row):
                        score += rule['weight']
                        reasons.append(rule['description'])
                    total_weight += rule['weight']
                
                # Normalize score
                if total_weight > 0:
                    score = score / total_weight
                
                results.append({
                    'score': score,
                    'reasons': reasons
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error applying expert rules: {str(e)}")
            return [{'score': 0, 'reasons': ['Error applying rules']}]
    
    def get_rule_explanations(self):
        """Get explanations for all rules"""
        return [
            {
                'name': rule['name'],
                'description': rule['description'],
                'weight': rule['weight']
            }
            for rule in self.rules
        ]
