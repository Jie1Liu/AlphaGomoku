import socket

HOST = ''
PORT = 10888
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
conn, addr = s.accept()
print('Client\'s Address:',addr)


while True:
    data2 = conn.recv(2048)
    print(data2.decode('utf-8'))
    data= input("输入：")
    conn.sendall(data.encode('utf-8'))

