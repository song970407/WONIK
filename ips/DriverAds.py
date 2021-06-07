import pyads


def logger(func):
    def wrappedFunc(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)

    return wrappedFunc


class DriverAds(object):
    PLCTYPE_BOOL = pyads.PLCTYPE_BOOL
    PLCTYPE_BYTE = pyads.PLCTYPE_BYTE
    PLCTYPE_REAL = pyads.PLCTYPE_REAL
    PLCTYPE_STRING = pyads.PLCTYPE_STRING

    def __del__(self):
        if self.plc is not None:
            self.plc.close()

    @logger
    def connectPlc(self, ip, port=pyads.PORT_TC2PLC1):
        # plc = pyads.Connection('5.10.100.107.1.1', pyads.PORT_TC2PLC1)
        self.plc = pyads.Connection(ip, port)
        self.plc.open()

    def isConnect(self):
        if self.plc is not None:
            return self.plc.is_open()

    @logger
    def disconnectPlc(self):
        if self.plc is not None:
            self.plc.close()

    def readByName(self, ChName, Type=PLCTYPE_BOOL):
        value = self.plc.read_by_name(ChName, Type)
        return value

    def writeByName(self, ChName, Type=PLCTYPE_BOOL, value=''):
        self.plc.write_by_name(ChName, value, Type)
