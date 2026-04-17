import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://192.168.0.164:5560")

sock.send_string("hello from test")
reply = sock.recv_string()
print(f"Reply: {reply}")

sock.close()
ctx.term()