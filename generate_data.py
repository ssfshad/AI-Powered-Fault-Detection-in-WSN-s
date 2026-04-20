import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_wsn_data(n_samples=5000):
    fault_types = ['Normal', 'Node_Failure', 'Data_Anomaly', 'Battery_Drain', 'Communication_Loss']
    weights     = [0.40, 0.15, 0.15, 0.15, 0.15]
    n_nodes     = 20
    data        = []

    for _ in range(n_samples):
        fault   = np.random.choice(fault_types, p=weights)
        node_id = np.random.randint(1, n_nodes + 1)

        if fault == 'Normal':
            temp          = np.random.normal(27, 3)
            humidity      = np.random.normal(55, 8)
            signal        = np.random.uniform(-60, -40)
            battery       = np.random.uniform(50, 100)
            packet_loss   = np.random.uniform(0, 5)
            response_time = np.random.normal(25, 8)

        elif fault == 'Node_Failure':
            temp          = np.random.uniform(0, 80)
            humidity      = np.random.uniform(0, 100)
            signal        = np.random.uniform(-90, -70)
            battery       = np.random.uniform(5, 80)
            packet_loss   = np.random.uniform(80, 100)
            response_time = np.random.uniform(500, 1000)

        elif fault == 'Data_Anomaly':
            temp     = np.random.choice([
                np.random.uniform(65, 100),
                np.random.uniform(-20, 0)
            ])
            humidity      = np.random.choice([
                np.random.uniform(90, 100),
                np.random.uniform(0, 5)
            ])
            signal        = np.random.uniform(-60, -40)
            battery       = np.random.uniform(40, 100)
            packet_loss   = np.random.uniform(0, 10)
            response_time = np.random.normal(30, 10)

        elif fault == 'Battery_Drain':
            temp          = np.random.normal(32, 4)
            humidity      = np.random.normal(55, 8)
            signal        = np.random.uniform(-70, -50)
            battery       = np.random.uniform(0, 15)
            packet_loss   = np.random.uniform(10, 30)
            response_time = np.random.uniform(100, 300)

        else:  # Communication_Loss
            temp          = np.random.normal(27, 3)
            humidity      = np.random.normal(55, 8)
            signal        = np.random.uniform(-95, -75)
            battery       = np.random.uniform(40, 100)
            packet_loss   = np.random.uniform(50, 90)
            response_time = np.random.uniform(200, 800)

        # Clamp to physical limits
        temp          = np.clip(temp + np.random.normal(0, 0.3),  -30,  110)
        humidity      = np.clip(humidity + np.random.normal(0, 0.3), 0,  100)
        battery       = np.clip(battery,   0, 100)
        packet_loss   = np.clip(packet_loss, 0, 100)
        response_time = np.clip(response_time, 1, 1000)

        data.append({
            'node_id':          node_id,
            'temperature':      round(temp, 2),
            'humidity':         round(humidity, 2),
            'signal_strength':  round(signal, 2),
            'battery_level':    round(battery, 2),
            'packet_loss_rate': round(packet_loss, 2),
            'response_time':    round(response_time, 2),
            'fault_type':       fault,
        })

    return pd.DataFrame(data)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_wsn_data(5000)
    df.to_csv('data/wsn_dataset.csv', index=False)
    print(f"Dataset saved -> data/wsn_dataset.csv  ({len(df)} rows)")
    print(df['fault_type'].value_counts().to_string())
