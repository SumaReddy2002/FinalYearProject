import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import winsound
import os

def main():
    st.title("Device Data")

    # Define the columns for plotting
    col1, col2, col3 = st.columns(3)

    # Use an infinite loop to constantly check for updates
    while True:
        # Load the data
        data = pd.read_csv('device_data.csv')
        st.table(data)
        # Create the plots
        for i, device in enumerate(data.columns):
            # Select the appropriate column for this plot
            if i % 3 == 0:
                col = col1
            elif i % 3 == 1:
                col = col2
            else:
                col = col3

            # Plot the data
            fig = plt.figure(figsize=(5, 5))
            plt.plot(data[device])
            plt.title(device)
            col.pyplot(fig)
            col.write(data[device].sum())

            # If the total of this column is above the threshold, play a beep sound
            if data[device].sum() > 10000:
                winsound.Beep(1000, 1000)  # Frequency 1000 Hz, Duration 1 second

        # Wait for 5 seconds
        time.sleep(3)
        
        # Rerun the app to get fresh data
        st.experimental_rerun()

if __name__ == "__main__":
    main()
