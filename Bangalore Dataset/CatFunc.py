def categorize_congestion(value):
    if value < 30:
        return 'Low'
    elif value < 70:
        return 'Medium'
    else:
        return 'High'

def categorize_signals(value):
    if value > 90:
        return 'Low'
    elif value > 75:
        return 'Medium'
    else:
        return 'High'

