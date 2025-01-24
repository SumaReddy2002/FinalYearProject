import threading
import os
# Define a function to be executed in the first thread
def thread1_function():
    # Code for thread 1
    os.system("streamlit run range.py")

# Define a function to be executed in the second thread
def thread2_function():
    # Code for thread 2
    os.system("streamlit run enter.py")

# Create the first thread
thread1 = threading.Thread(target=thread1_function)

# Create the second thread
thread2 = threading.Thread(target=thread2_function)

thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()