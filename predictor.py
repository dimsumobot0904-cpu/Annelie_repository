import joblib
import numpy as np
import tensorflow as tf

# Load models once upon server startup
try:
    scaler = joblib.load('scaler.pkl')
    lstm_temp = tf.keras.models.load_model('lstm_temp.keras')
    svm_temp = joblib.load('svm_temp.pkl')

    lstm_ph = tf.keras.models.load_model('lstm_ph.keras')
    svm_ph = joblib.load('svm_ph.pkl')

    lstm_do = tf.keras.models.load_model('lstm_do.keras')
    svm_do = joblib.load('svm_do.pkl')
    print("✅ All AI models loaded successfully in predictor.py!")
except Exception as e:
    print(f"❌ Error loading models in predictor.py: {e}")
    scaler = lstm_temp = svm_temp = lstm_ph = svm_ph = lstm_do = svm_do = None


def _unscale(value, index):
    """Helper function to reverse scaling for 3 parameters"""
    dummy = np.zeros((1, 3))
    dummy[0, index] = value
    return scaler.inverse_transform(dummy)[0, index]


def predict_water_quality(last_10_readings):
    """
    Input: list or numpy array of shape (10, 3) -> [[Temp_C, pH, DO_ppm], ...]
    Returns: Python dictionary containing LSTM forecast & SVM status
    """
    if scaler is None or lstm_temp is None:
        raise ValueError("Models are not loaded properly.")

    data_arr = np.array(last_10_readings)

    # Scale input data
    scaled_input = scaler.transform(data_arr)

    # Reshape for LSTM (1, 10, 3) and SVM (1, 30)
    lstm_in = np.expand_dims(scaled_input, axis=0)
    svm_in = lstm_in.reshape(1, -1)

    # Generate Predictions
    pred_lstm_temp = _unscale(lstm_temp.predict(lstm_in, verbose=0)[0][0], 0)
    pred_svm_temp = _unscale(svm_temp.predict(svm_in)[0], 0)

    pred_lstm_ph = _unscale(lstm_ph.predict(lstm_in, verbose=0)[0][0], 1)
    pred_svm_ph = _unscale(svm_ph.predict(svm_in)[0], 1)

    pred_lstm_do = _unscale(lstm_do.predict(lstm_in, verbose=0)[0][0], 2)
    pred_svm_do = _unscale(svm_do.predict(svm_in)[0], 2)

    # Evaluate Safe / Unsafe Status (SVM)
    svm_temp_status = 'Safe ✅' if 28.0 <= pred_svm_temp <= 31.0 else 'Unsafe ⚠️'
    svm_ph_status = 'Safe ✅' if 7.5 <= pred_svm_ph <= 8.5 else 'Unsafe ⚠️'
    svm_do_status = 'Safe ✅' if pred_svm_do >= 5.0 else 'Unsafe ⚠️'

    return {
        'temp': {
            'lstm_prediction': round(float(pred_lstm_temp), 2),
            'svm_status': svm_temp_status,
        },
        'ph': {
            'lstm_prediction': round(float(pred_lstm_ph), 2),
            'svm_status': svm_ph_status,
        },
        'do': {
            'lstm_prediction': round(float(pred_lstm_do), 2),
            'svm_status': svm_do_status,
        },
    }


if __name__ == '__main__':
    import pandas as pd
    print('🔄 Testing predictor.py with CSV data...')
    try:
        df = pd.read_csv('5.-datasetsfrom-feb-15-mar15.csv')
        sample_10_readings = df[['Temp_C', 'pH', 'DO_ppm']].iloc[-10:].values
        test_result = predict_water_quality(sample_10_readings)
        print('\n✅ TEST SUCCESSFUL! AI Prediction Output:')
        print(test_result)
    except Exception as e:
        print(f"Test failed: {e}")