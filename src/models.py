from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

def get_gradient_boosting():
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    
if __name__ == "__main__":
    print(get_random_forest())
    print(get_gradient_boosting())