import numpy as np

def calculate_precision_recall(signature_image1, signature_image2):
    # Dummy values (replace with actual signature verification results)
    true_positive = np.random.randint(0, 100)
    false_positive = np.random.randint(0, 100)
    false_negative = np.random.randint(0, 100)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f"precision: {precision:.2f}, recall: {recall:.2f}, f1_score: {f1_score:.2f}"