import streamlit as st
import os
import threading

# Define a function to be executed in the first thread
def thread1_function():
    # Code for thread 1
    os.system("python interface.py")

# Define a function to be executed in the second thread
def thread2_function():
    # Code for thread 2
    os.system("streamlit run app.py")
 
# Create the first thread
thread1 = threading.Thread(target=thread1_function)

# Create the second thread
thread2 = threading.Thread(target=thread2_function)
# Title of the application
st.title('Electricity Home Simualtion Configuration')

# Input for the number of boxes
num_boxes = st.number_input("Enter the number of boxes", min_value=1, max_value=10, step=1)

# A list of available devices
device_names = ['bulb', 'tube', 'fan', 'oven', 'ac', 'lamp', 'cooler', 'tv', 'fridge']

# A dict to store the devices for each box
devices_in_boxes = {}

for i in range(num_boxes):
    devices_in_boxes[f'box{i+1}'] = st.multiselect(f'Select devices for Box {i+1}', device_names)

# When the 'Save configuration' button is pressed
if st.button('Save configuration'):
    # Save the number of boxes to a text file
    with open('num_boxes.txt', 'w') as f:
        f.write(str(num_boxes))

    # Save the devices for each box to separate text files
    for box, devices in devices_in_boxes.items():
        # Write the devices in order, replace non-selected devices with 'nil'
        with open(f'{box}.txt', 'w') as f:
            f.write(','.join(device if device in devices else 'nil' for device in device_names))

    st.success('Configuration saved successfully!')
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
    
