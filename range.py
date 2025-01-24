import streamlit as st
import json
import os
import threading
import os
# Define a function to be executed in the first thread
def thread1_function():
    # Code for thread 1
    os.system("python data.py")

# Define a function to be executed in the second thread
def thread2_function():
    # Code for thread 2
    os.system("streamlit run simu.py")
def thread3_function():
    # Code for thread 2
    os.system("python cam.py")
# Function to load device state information from a file
def load_state():
    with open('device_state.json', 'r') as f:
        return json.load(f)

# Function to save device range information to a file
def save_range(range_info):
    with open('range.json', 'w') as f:
        json.dump(range_info, f)

def main():
    st.title("Set Device Ranges")

    state = load_state()  # Initial state
    device_ranges = {}

    for box_num, box in enumerate(state):
        for device_name, device_state in box.items():
            # Ensure the key for each slider is unique by including box_num and device_name in the key
            key = f"{box_num}-{device_name}"
            device_range = st.slider(f"Set the range for {device_name.capitalize()}", 0, 100, (0, 100), key=key)
            device_ranges[device_name] = device_range

    if st.button('Save Ranges'):
        save_range(device_ranges)
        st.success('Device ranges saved successfully!')
       
        thread1 = threading.Thread(target=thread1_function)

# Create the second thread
        thread2 = threading.Thread(target=thread2_function)
        thread3 = threading.Thread(target=thread3_function)

        thread1.start()
        thread2.start()
        thread3.start()
        # Wait for both threads to finish
        thread1.join()
        thread2.join()
        thread3.join()

if __name__ == "__main__":
    main()
