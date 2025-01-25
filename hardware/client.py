import socket
import subprocess

client_socket = socket.socket()
client_socket.connect(('192.168.233.149', 8000))

# Simplified ffplay options for better compatibility
cmd = ['ffplay', 
       '-fflags', 'nobuffer',
       '-flags', 'low_delay',
       '-framedrop',
       '-i', 'pipe:0']

player = subprocess.Popen(cmd, stdin=subprocess.PIPE)

try:
    buffer_size = 4096
    while True:
        data = client_socket.recv(buffer_size)
        if not data:
            break
        try:
            player.stdin.write(data)
            player.stdin.flush()
        except BrokenPipeError:
            break
except KeyboardInterrupt:
    print("\nStopping stream...")
except Exception as e:
    print(f"Error: {e}")
finally:
    player.terminate()
    client_socket.close()

