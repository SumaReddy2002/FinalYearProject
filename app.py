import streamlit as st
import json
import time
import os

# Function to load device state information from a file
def load_state():
    with open('device_state.json', 'r') as f:
        return json.load(f)

# Streamlit application
def main():
    state = load_state()  # Initial state

    # Set up Streamlit page
    st.title("Device State")

    # Display initial state
    for box_num, box in enumerate(state):
        st.header(f"Box {box_num + 1}")
        for device_name, device_state in box.items():
            st.text(f"{device_name.capitalize()}: {'ON' if device_state else 'OFF'}")

    # Update page with new state every second
    while True:
        time.sleep(1)
        new_state = load_state()
        if new_state != state:
            state = new_state
            st.experimental_rerun()  # Redraw page

if __name__ == "__main__":
    
    main()
