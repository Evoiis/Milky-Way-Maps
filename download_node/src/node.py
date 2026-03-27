import zmq
import star_data_pb2

# sd = star_data_pb2.StarData()
# sd.pos_x = 10.12
# print(sd.SerializeToString())

# k = star_data_pb2.StarData()
# k.ParseFromString(sd.SerializeToString())
# print(k.pos_x)

class DownloadNode():

    def __init__(self, query_wrapper):
        self.zmq_context = zmq.Context()
        self.socket = zmq_context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        self.query_wrapper = query_wrapper

    def loop(self):
        try:
            while True:

                # Wait for msg
                message = self.socket.recv()

                print("Received message: ", message)

                self.socket.send("Reply")

        except KeyboardInterrupt:
            pass

