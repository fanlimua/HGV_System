#!/usr/bin/env python

import cv2
import socket
import pickle
import struct
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Remote Camera Server')
    parser.add_argument('--port', type=int, default=8089, 
                        help='Port to use for streaming')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera device index')
    parser.add_argument('--quality', type=int, default=50, 
                        help='JPEG compression quality (0-100)')
    parser.add_argument('--resolution', type=str, default='640x480', 
                        help='Camera resolution in format WIDTHxHEIGHT')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Create a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    print(f'HOST IP: {host_ip}')
    
    socket_address = ('0.0.0.0', args.port)
    server_socket.bind(socket_address)
    server_socket.listen(5)
    print(f"Listening on {socket_address}")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Accept connections
    while True:
        client_socket, addr = server_socket.accept()
        print(f'Connection from: {addr}')
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break
                
                # Compress frame to reduce network load
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
                result, frame_encoded = cv2.imencode('.jpg', frame, encode_param)
                
                data = pickle.dumps(frame_encoded)
                message_size = struct.pack("L", len(data))
                
                try:
                    client_socket.sendall(message_size + data)
                    
                    # Optional: Show the frame locally
                    cv2.imshow('Remote Camera Stream', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                        break
                        
                except (ConnectionResetError, BrokenPipeError):
                    print("Connection lost")
                    break
                    
        finally:
            client_socket.close()
            print(f'Connection with {addr} closed')
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    server_socket.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Server stopped by user") 