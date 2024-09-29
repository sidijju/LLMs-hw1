import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Ridge

def f(x):
    return x + np.sin(1.5 * x)

def sample_fx_data(x, noise_std=0.3):
    return f(x) + np.random.randn(*x.shape) * noise_std

def poly_fit_transform(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)

def part2():
    x = 10 * np.random.random(20)
    x = np.sort(x)
    y = sample_fx_data(x)
    x_smooth = np.linspace(0, 10, 100)
    fx = f(x_smooth)
    plt.scatter(x, y, label='y', color='r')
    plt.plot(x_smooth, fx, label='f(x)')
    plt.legend()
    plt.savefig('q1-part2.png')
    return x, y, x_smooth, fx

def part3(x, y, x_smooth, fx):
    g1 = poly_fit_transform(x, y, 1)
    g3 = poly_fit_transform(x, y, 3)
    g10 = poly_fit_transform(x, y, 10)
    plt.cla()
    plt.plot(x_smooth, fx, label='f(x)')
    plt.plot(x, g1, label='g1')
    plt.plot(x, g3, label='g3')
    plt.plot(x, g10, label='g10')
    plt.scatter(x, y, label='y', color='r')
    plt.legend()
    plt.savefig('q1-part3.png')

def part4_part5():
    NUM_DATASETS = 100
    NUM_OBSERVATIONS = 50
    N_TRAIN = int(NUM_OBSERVATIONS * 0.8)
    HIGHEST_DEGREE = 15

    # fixed x across datasets
    x = 10 * np.random.random(NUM_OBSERVATIONS)
    x_train = x[:N_TRAIN]
    x_test = x[N_TRAIN:]

    test_predictions = defaultdict(list)
    test_errors = defaultdict(list)

    for _ in range(NUM_DATASETS):
        y_train, y_test = sample_fx_data(x_train), sample_fx_data(x_test)

        for degree in range(1, HIGHEST_DEGREE+1):
            coeffs = np.polyfit(x_train, y_train, degree)
            prediction = np.polyval(coeffs, x_test)
            test_predictions[degree].append(prediction)
            test_errors[degree].append((prediction - y_test)**2)

    squared_biases = []
    variances = []
    errors = []

    def calculate_squared_bias(test_prediction):
        test_prediction = np.array(test_prediction)
        average_model_prediction = test_prediction.mean(0)
        return np.mean((average_model_prediction - f(x_test)) ** 2)

    def calculate_variance(test_prediction):
        test_prediction = np.array(test_prediction)
        average_model_prediction = test_prediction.mean(0)
        return np.mean((test_prediction - average_model_prediction) ** 2)

    for degree in range(1, HIGHEST_DEGREE+1):
        squared_biases.append(calculate_squared_bias(test_predictions[degree]))
        variances.append(calculate_variance(test_predictions[degree]))
        errors.append(np.mean(test_errors[degree]))

    degrees = list(range(1, HIGHEST_DEGREE+1))
    best_model_degree = np.argmin(errors) + 1

    plt.cla()
    plt.plot(degrees, squared_biases, label='squared bias')
    plt.plot(degrees, variances, label='variance')
    plt.plot(degrees, errors, label='error')
    plt.axvline(best_model_degree, linestyle='--', color='black', label=f'best model(degree={best_model_degree})')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('q1-part4.png')

    #### Part 5 ####

    model_degree = 10

    regularized_predictions = []
    regularized_errors = []

    for _ in range(NUM_DATASETS):
        y_train, y_test = sample_fx_data(x_train), sample_fx_data(x_test)
        x_train_polynomial = np.vander(x_train, N=model_degree+1, increasing=True)        
        ridge_model = Ridge().fit(x_train_polynomial, y_train)
        x_test_polynomial = np.vander(x_test, N=model_degree+1, increasing=True)
        prediction = ridge_model.predict(x_test_polynomial)
        regularized_predictions.append(prediction)
        regularized_errors.append((prediction - y_test)**2)

    regularized_squared_bias = calculate_squared_bias(regularized_predictions)
    regularized_variance = calculate_variance(regularized_predictions)
    regularized_error = np.mean(regularized_errors)

    print("Unregularized stats")
    print("squared bias", squared_biases[model_degree-1])
    print("variance", variances[model_degree-1])
    print("error", errors[model_degree-1])
    print()

    print("Regularized stats")
    print("squared bias", regularized_squared_bias)
    print("variance", regularized_variance)
    print("error", regularized_error)

# def part5(x_train, x_test):
#     y_train, y_test = sample_fx_data(x_train), sample_fx_data(x_test)
#     ridge_model = Ridge.fit(x_train, y_train)
#     prediction = ridge_model.predict(x_test)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(123)

    part3(*part2())
    part4_part5()
    