"""Quick check of model classes and predictions"""
import pickle
import numpy as np

# Load BTC model
with open('models/saved_models/BTC_USDT/lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('BTC Model classes:', model.classes_)

# Test random inputs
print('\nRandom prediction tests:')
for i in range(10):
    test = np.random.randn(1, 21)
    pred = model.predict(test)
    proba = model.predict_proba(test)[0]
    print(f'  Test {i+1}: class={model.classes_[pred[0]]}, probs=[{proba[0]:.2f}, {proba[1]:.2f}, {proba[2]:.2f}]')

# Load ETH model
with open('models/saved_models/ETH_USDT/lstm_model.pkl', 'rb') as f:
    eth_model = pickle.load(f)

print('\nETH Model classes:', eth_model.classes_)
