import cv2
import socket
import pickle
import struct

# Raspberry Pi's IP address and port for streaming
rasp_ip = 'your_raspberry_pi_ip'
port = 5001

# Create a socket connection to the Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((rasp_ip, port))

# Create an OpenCV window for displaying the stream
cv2.namedWindow("Webcam Stream", cv2.WINDOW_NORMAL)

while True:
    try:
        # Receive the message size and frame data from the Raspberry Pi
        data = b""
        payload_size = struct.calcsize("L")
        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Unpickle and display the frame
        frame = pickle.loads(frame_data)
        cv2.imshow("Webcam Stream", frame)
        cv2.waitKey(1)

    except Exception as e:
        print(f"Error: {str(e)}")
        break

# Close the connection and destroy the OpenCV window
client_socket.close()
cv2.destroyAllWindows()
