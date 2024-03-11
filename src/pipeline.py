import argparse



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def irt_model(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def neg_log_likelihood(params, data):
    a, b, c = params
    theta = data[:, 0]
    correct = data[:, 1]
    p = irt_model(theta, a, b, c)
    log_likelihood = correct * np.log(p) + (1 - correct) * np.log(1 - p)
    return -np.sum(log_likelihood)

def generate_irt_curves(models, data):
    for model, model_data in zip(models, data):
        initial_params = [1, 0, 0.25]  # Initial parameter values [a, b, c]
        result = minimize(neg_log_likelihood, initial_params, args=(model_data,), method='Nelder-Mead')
        a, b, c = result.x

        theta_range = np.linspace(-4, 4, 100)
        prob_correct = irt_model(theta_range, a, b, c)

        plt.plot(theta_range, prob_correct, label=model)

    plt.xlabel('Ability (theta)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Item Response Theory (IRT) Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage





if __name__ == "__main__":
    # code to run item response theory for LLMs to predict student performance
    argparser = argparse.ArgumentParser(description="Run IRT for LLMs")
    argparser.add_argument("--data", type=str, help="Path to data file")

    # Step 1: Get language models as students to predict student performance
    # Step 2: Get student performance data
    # Step 3: Run IRT for LLMs
    # Step 4: Output results
    # Step 5: Evaluate results


    models = ['Model A', 'Model B', 'Model C']
    data = [
        np.array([[1.5, 1], [0.8, 1], [-0.5, 0], [-1.2, 0], [0.3, 1]]),
        np.array([[2.0, 1], [1.2, 1], [0.2, 1], [-0.8, 0], [-1.5, 0]]),
        np.array([[1.8, 1], [1.0, 1], [-0.2, 1], [-1.0, 0], [-1.8, 0]])
    ]

    generate_irt_curves(models, data)

