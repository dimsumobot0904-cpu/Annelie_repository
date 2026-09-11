from collections import deque
import datetime
import json
import sqlite3
import threading
import time
from flask import Flask, jsonify, redirect, render_template, request, session
import paho.mqtt.client as mqtt

# I-import ang AI prediction function mula sa iyong predictor.py
from predictor import predict_water_quality

app = Flask(__name__)
app.secret_key = 'hatchguard_secret_key_123'
DB_NAME = 'ulang_data.db'

# MQTT Configuration
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
TOPIC_SENSOR = 'hatchguard/sensors'
TOPIC_RELAY = 'hatchguard/relay'

latest_sensor_data = {
    'temp': '29.1',
    'ph': '7.7',
    'do': '5.23',
    'salinity': '12',
    'ammonia': '0.02',
    'nitrate': '12.0',
    'nitrite': '0.04',
    'turbidity': '18.3',
}

# Buffer para sa huling 10 readings na kailangan ng AI (LSTM/SVM)
sensor_history = deque(maxlen=10)

# RELAY STATES
relay_states = {'do': 1, 'ph': 1, 'temp': 1, 'sal': 1, 'feeder': 0}

# DEFAULT DYNAMIC SCHEDULES
DEFAULT_SCHEDULES = ['06:00', '12:00', '18:00', '00:00']
dynamic_schedules = DEFAULT_SCHEDULES


# --- DATABASE SETUP & HELPERS ---
def init_db():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()

  # Table para sa History Logs ng Sensors (Para hindi mawala ang readings)
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dissolved_oxygen REAL, ph REAL, temperature REAL, 
            salinity REAL, ammonia REAL, nitrate_nitrite REAL, turbidity REAL
        )
    ''')

  # Table para sa System Configurations (Feeding Schedules, atbp. para hindi mag-reset)
  cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

  conn.commit()
  conn.close()
  load_schedules_from_db()


def save_schedule_to_db(schedules_list):
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  schedules_str = json.dumps(schedules_list)
  cursor.execute(
      'REPLACE INTO system_config (key, value) VALUES (?, ?)',
      ('dynamic_schedules', schedules_str),
  )
  conn.commit()
  conn.close()


def load_schedules_from_db():
  global dynamic_schedules
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      'SELECT value FROM system_config WHERE key = ?', ('dynamic_schedules',)
  )
  row = cursor.fetchone()
  conn.close()

  if row:
    try:
      dynamic_schedules = json.loads(row[0])
    except:
      dynamic_schedules = DEFAULT_SCHEDULES
  else:
    dynamic_schedules = DEFAULT_SCHEDULES
    save_schedule_to_db(DEFAULT_SCHEDULES)


# --- BACKGROUND TIMER PARA SA SCHEDULED FEEDING ---
def check_schedule():
  global dynamic_schedules
  while True:
    current_time = datetime.datetime.now().strftime('%H:%M')

    if current_time in dynamic_schedules:
      payload = json.dumps({'relay': 'feeder', 'state': 1})
      mqtt_client.publish(TOPIC_RELAY, payload)
      print(f'\n⏰ Scheduled Feeding Triggered at {current_time}!\n')
      time.sleep(61)

    time.sleep(10)


# --- MQTT CALLBACK KUNG MAY DATA MULA ESP32 SENSORS ---
def on_message(client, userdata, msg):
  global latest_sensor_data, sensor_history
  try:
    payload = msg.payload.decode('utf-8')
    data = json.loads(payload)

    # 1. Update live readings mula sa aktwal na sensors
    if 'temp' in data:
      latest_sensor_data['temp'] = str(data['temp'])
    if 'ph' in data:
      latest_sensor_data['ph'] = str(data['ph'])
    if 'do' in data:
      latest_sensor_data['do'] = str(data['do'])
    if 'ammonia' in data:
      latest_sensor_data['ammonia'] = str(data['ammonia'])
    if 'nitrate' in data:
      latest_sensor_data['nitrate'] = str(data['nitrate'])
    if 'nitrite' in data:
      latest_sensor_data['nitrite'] = str(data['nitrite'])
    if 'turbidity' in data:
      latest_sensor_data['turbidity'] = str(data['turbidity'])

    # 2. I-save agad sa SQLite Database para sa permanent history log
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        '''
            INSERT INTO sensor_readings (dissolved_oxygen, ph, temperature, salinity, ammonia, nitrate_nitrite, turbidity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            float(latest_sensor_data['do']),
            float(latest_sensor_data['ph']),
            float(latest_sensor_data['temp']),
            float(latest_sensor_data['salinity']),
            float(latest_sensor_data['ammonia']),
            float(latest_sensor_data['nitrate']),
            float(latest_sensor_data['turbidity']),
        ),
    )
    conn.commit()
    conn.close()

    # 3. I-push sa history buffer (Temp, pH, DO) para sa AI Prediction
    current_features = [
        float(latest_sensor_data['temp']),
        float(latest_sensor_data['ph']),
        float(latest_sensor_data['do']),
    ]
    sensor_history.append(current_features)

    print(f'Sensor Data Received & Saved to DB: {latest_sensor_data}')
  except Exception as e:
    print(f'Error sa pagbasa ng MQTT message: {e}')


# Setup MQTT Client
try:
  mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
except AttributeError:
  mqtt_client = mqtt.Client()

mqtt_client.on_message = on_message


def start_mqtt():
  try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.subscribe(TOPIC_SENSOR)
    mqtt_client.loop_start()
    print('Konektado na sa MQTT Broker!')
  except Exception as e:
    print(f'MQTT Connection Error: {e}')


# ========================================================
# FLASK WEB ROUTES & ENDPOINTS
# ========================================================

USER_DATA = {'dimsumobot': 'strongpassword'}


@app.route('/login', methods=['GET', 'POST'])
def login():
  error = None
  if request.method == 'POST':
    if (
        request.form.get('username') in USER_DATA
        and USER_DATA[request.form.get('username')]
        == request.form.get('password')
    ):
      session['logged_in'] = True
      return redirect(url_for('dashboard'))
    error = 'Invalid Username or Password!'
  return render_template('login.html', error=error)


@app.route('/logout')
def logout():
  session.clear()
  return redirect(url_for('login'))


@app.route('/')
@app.route('/dashboard')
def dashboard():
  if not session.get('logged_in'):
    return redirect(url_for('login'))
  return render_template('index.html')


@app.route('/sensors')
def sensors():
  if not session.get('logged_in'):
    return redirect(url_for('login'))
  return render_template('sensors.html')


@app.route('/feeding')
def feeding():
  if not session.get('logged_in'):
    return redirect(url_for('login'))
  return render_template('feeding.html')


@app.route('/get_data')
def get_data():
  if not session.get('logged_in'):
    return jsonify({'error': 'Unauthorized'}), 401
  return jsonify(latest_sensor_data)


# --- AI PREDICTION ROUTE (Kumukuha sa actual sensor data buffer) ---
@app.route('/predict_ai')
def predict_ai():
  if not session.get('logged_in'):
    return jsonify({'error': 'Unauthorized'}), 401

  try:
    current_features = [
        float(latest_sensor_data['temp']),
        float(latest_sensor_data['ph']),
        float(latest_sensor_data['do']),
    ]

    # Siguraduhing may sapat na 10 readings sa buffer bago patakbuhin ang AI
    while len(sensor_history) < 10:
      sensor_history.append(current_features)

    last_10 = list(sensor_history)

    # Tawagin ang iyong totoong AI model function mula sa predictor.py
    result = predict_water_quality(last_10)

    # Suriin kung Safe ang lahat ng parameters base sa SVM results
    all_safe = all(
        'Safe' in result[key]['svm_status'] for key in ['temp', 'ph', 'do']
    )

    return jsonify({
        'status': 'success',
        'forecast_temp': result['temp']['lstm_prediction'],
        'forecast_ph': result['ph']['lstm_prediction'],
        'forecast_do': result['do']['lstm_prediction'],
        'temp_status': result['temp']['svm_status'],
        'ph_status': result['ph']['svm_status'],
        'do_status': result['do']['svm_status'],
        'svm_status': 'SAFE' if all_safe else 'UNSAFE',
        'risk_color': '#22c55e' if all_safe else '#ef4444',
    })

  except Exception as e:
    print(f'AI Prediction Error: {e}')
    return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/update_schedule', methods=['POST'])
def update_schedule():
  global dynamic_schedules
  if not session.get('logged_in'):
    return jsonify({'error': 'Unauthorized'}), 401

  data = request.get_json()
  new_schedules = data.get('schedules')

  if new_schedules and isinstance(new_schedules, list):
    dynamic_schedules = new_schedules
    save_schedule_to_db(dynamic_schedules)
    print(f'\n[!] Na-save ang feeding schedule sa DB: {dynamic_schedules}\n')
    return jsonify({'status': 'success', 'schedules': dynamic_schedules})

  return jsonify({'status': 'error', 'message': 'Invalid data format'}), 400


@app.route('/control_relay', methods=['POST'])
def control_relay():
  if not session.get('logged_in'):
    return jsonify({'error': 'Unauthorized'}), 401

  data = request.get_json()
  relay = data.get('relay')
  state = data.get('state')

  if relay in relay_states:
    relay_states[relay] = state
    payload = json.dumps({'relay': relay, 'state': state})
    mqtt_client.publish(TOPIC_RELAY, payload)
    return jsonify({'status': 'success', 'relay': relay, 'state': state})

  return jsonify({'status': 'error', 'message': 'Invalid relay'}), 400


if __name__ == '__main__':
  init_db()
  start_mqtt()
  threading.Thread(target=check_schedule, daemon=True).start()
  app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)