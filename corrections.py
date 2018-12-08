import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import extract_features # make sure features.py is in the same directory
import activity_recognition
from util import reorient, reset_vars
from scipy.ndimage.interpolation import shift

# TODO: Replace the string with your user ID
user_id = "fifa3"
chosen_activity = 'sitting'
x_values = np.zeros(250)
y_values = np.zeros(250)
z_values = np.zeros(250)
t_values = np.zeros(250)
'''
***   Final Project code goes here!  ***
*** Using sitting as a default exercise ***
'''
count = 0

def correct_motion(activity):
    global count
    # if activity == chosen_activity:
    if count % 4 == 0:
        print("You are " + activity)
        # data = analyse_data(activity)
        # provide_feedback(data, activity)
    count+= 1

'''
Used to find most wrong dimension that we can then use in the provide feedback method to
provide feedback of the correct nature
'''
def analyse_data(activity):
    global x_values
    global y_values
    global z_values
    global t_values

    if activity == chosen_activity:
        axis = 'none'
        print("We will now analyse your form and attempt to provide feedback")

        '''
        Here do data anlysis on x,y,z axis's to determine which is most incorrect for chosen_activity
        '''

    return axis

'''
Provides feedback to the user in the form of text to help them improve exercise form
The text per axis will change depending on the exercise
'''
def provide_feedback(axis, activity):
    if activity == chosen_activity:
        if(axis == 'x'):
            print('')
        elif(axis =='y'):
            print('')
        elif(axis == 'z'):
            print('')
        elif(axis =='none'):
            print("We detect nothing wrong with your form keep up the good work!")



'''
Code for connecting to phone
'''
def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.

    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip().decode('ascii')
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id).encode('utf-8'))

    try:
        message = sock.recv(256).strip().decode('ascii')
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")

    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception(
            "Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))

    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception(
            "Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))

receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.connect(("none.cs.umass.edu", 8888))
# ensures that after 1 second, a keyboard interrupt will close
receive_socket.settimeout(1.0)
activity = activity_recognition
msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

try:
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)

    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()

    previous_json = ''

    sensor_data = []
    window_size = 25  # ~1 sec assuming 25 Hz sampling rate
    step_size = 25  # no overlap
    index = 0  # to keep track of how many samples we have buffered so far
    reset_vars()  # resets orientation variables

    while True:
        try:
            message = receive_socket.recv(1024).strip().decode('ascii')
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = ''  # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_ACCEL"):
                    t = data['data']['t']
                    x = data['data']['x']
                    y = data['data']['y']
                    z = data['data']['z']

                    # x_values = shift(x_values, cval=1)
                    # x_values[0]= x
                    # y_values = shift(y_values, cval=1)
                    # y_values[0] = y
                    # z_values = shift(y_values, cval=1)
                    # z_values[0] = z
                    # t_values = shift(t_values, cval=1)
                    # t_values[0] = t


                    sensor_data.append(reorient(x, y, z))
                    index += 1
                    # make sure we have exactly window_size data points :
                    while len(sensor_data) > window_size:
                        sensor_data.pop(0)

                    if (index >= step_size and len(sensor_data) == window_size):
                        t = threading.Thread(target=activity.predict, args=(np.asarray(sensor_data[:]),))
                        correct_motion(activity.activity)
                        t.start()
                        index = 0

            sys.stdout.flush()
        except KeyboardInterrupt:
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely
                print(e)
            pass
except KeyboardInterrupt:
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Qutting...")
finally:
    print('closing socket for receiving data')
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()