import socket
import struct
import numpy as np
import cv2
import sys
from hand_control_remote import process_remote_frame, cleanup

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# Configuration
LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5623

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((LISTEN_IP, LISTEN_PORT))
server_socket.listen(1)
print(f"[INFO] Waiting for connection on {LISTEN_IP}:{LISTEN_PORT} ...")
conn, addr = server_socket.accept()
print(f"[INFO] Connected by {addr}")
print("[INFO] Waiting video feed...")

try:
    while True:
        packed_len = recvall(conn, 4)
        if not packed_len:
            break
        msg_len = struct.unpack('>L', packed_len)[0]
        data = recvall(conn, msg_len)
        if not data:
            break
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Process the received frame using hand_control_remote
        if not process_remote_frame(img):
            break
finally:
    conn.close()
    server_socket.close()
    cleanup()
