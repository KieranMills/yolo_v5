import socket
import time
import struct
from pynput import mouse
from pynput.mouse import Button, Controller
print('server initiated')
mouse = Controller()

HEADERSIZE = 10

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 80))
s.listen(5)
st = struct.Struct('I I')
while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()


    print(f"Connection from {address} has been established.")
    print(address)
    print('Flynn Has Arrived :O')

    msg = "Welcome to the server!"
    #msg = f"{len(msg):<{HEADERSIZE}}"+msg

#    clientsocket.send(bytes(msg,"utf-8"))




    while True:
        msg = str(mouse.position)
        print(msg)
        msg2 = msg.ljust(14)
        time.sleep(0.01)
        #print(bytes(msg2,"utf-8"))
        clientsocket.send(bytes(st.pack(*mouse.position)))
        #clientsocket.send(bytes(msg,"utf-8"))
        #clientsocket.close()
