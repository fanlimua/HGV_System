import cv2
import socket
import struct

# Configuration
SERVER_IP = '172.20.10.7'  # <-- Change to your server's LAN IP
SERVER_PORT = 5623

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, img_encoded = cv2.imencode('.jpg', frame)
        data = img_encoded.tobytes()
        # Send length first
        client_socket.sendall(struct.pack('>L', len(data)) + data)
        # Optional: show what is being sent
        cv2.imshow('Sending', frame)
        if cv2.waitKey(1) == 27:
            break
finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
