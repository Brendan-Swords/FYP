import grpc
import diambra.engine.interface_pb2_grpc as service
import diambra.engine.interface_pb2 as model
from diambra.engine.interface_pb2 import SpaceTypes
from diambra.engine.interface_pb2 import RamStatesCategories as Roles

class Client(service.EnvServerStub):
    """DIAMBRA Arena Engine API Client"""

    def __init__(self, address, timeout=60):
        self.channel = grpc.insecure_channel(address)
        super().__init__(self.channel)
        grpc.channel_ready_future(self.channel).result(timeout=timeout)
