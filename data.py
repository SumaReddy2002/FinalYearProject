import pandas as pd
import json
import random
import time

# Function to load device state information from a file
def load_state():
    with open('device_state.json', 'r') as f:
        return json.load(f)

# Function to load device range information from a file
def load_range():
    with open('range.json', 'r') as f:
        return json.load(f)

def main():
    
    rows = []
    while True: 
        state = load_state()  # Load device state
        ranges = load_range()  # Load device ranges

        devices = [device for box in state for device in box.keys()]
         # This loop runs indefinitely
        row = {}
        for device in devices:
            device_state = any(box[device] for box in state if device in box)
            row[device] = random.randint(*ranges[device]) if device_state else 0
        
        rows.append(row)

        data = pd.DataFrame(rows)
        data.to_csv('device_data.csv', index=False)  # Save DataFrame to a CSV file

        time.sleep(1)  # 
if __name__ == "__main__":
    main()
