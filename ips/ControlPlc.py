import threading
import time
import enum
from ips.DriverAds import DriverAds
import numpy as np
#from ips.TraceLog import TraceLog

controlTcMax = 40
glassTcMax = 140
arrayZoneMax = 70


class EnumProcessingState(enum.Enum):
    IDLE = 0
    PROCESSING = 1


class EnumReadyState(enum.Enum):
    NG = 0
    OK = 1


class ControlPlc(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, name='Thread')

        self.ads = DriverAds()
        self.processing = False
        self.ready = True

        self.rcp_name = ''
        ''' Current Recipe Name '''
        self.step_name = ''
        ''' Current Step Name '''
        self.step_time = 0
        ''' Current Step Time '''
        self.control_tc = 0
        self.glass_tc = 0
        self.heater_sp = 0
        self.heater_sv = 0

        '''
        self.preStepTime = 0
        self.preStepName = ''

        self.step_log = TraceLog()
        '''

    def __del__(self):
        if self.ads is not None:
            self.ads.disconnectPlc()

    def connect_plc(self):
        self.ads.connectPlc('5.10.100.101.1.1', 801)

    def check_processing(self):
        value = self.ads.readByName('.mDX_INF_State_Process', self.ads.PLCTYPE_BYTE)
        if value == 0:
            return EnumProcessingState.IDLE
        else:
            return EnumProcessingState.PROCESSING

    def check_ready(self):
        value = self.ads.readByName('.mDX_INF_State_Ready', self.ads.PLCTYPE_BYTE)
        if value == 0:
            return EnumReadyState.NG
        else:
            return EnumReadyState.OK

    def set_heater(self, value):
        self.ads.writeByName('.mAO_Heater_SP', self.ads.PLCTYPE_REAL * arrayZoneMax, value)  # set temp
        self.ads.writeByName('.mAO_Heater_Ramp', self.ads.PLCTYPE_REAL * arrayZoneMax,
                             np.zeros(arrayZoneMax, ))
        self.ads.writeByName('.mAO_Heater_PowerLimit', self.ads.PLCTYPE_REAL * arrayZoneMax,
                             np.full((arrayZoneMax,), 80))
        self.ads.writeByName('.mDO_Heater_Command', self.ads.PLCTYPE_BYTE, 1)

    def reset_heater(self):
        self.ads.writeByName('.mAO_Heater_SP', self.ads.PLCTYPE_REAL * arrayZoneMax,
                             np.zeros(arrayZoneMax, ))
        self.ads.writeByName('.mAO_Heater_Ramp', self.ads.PLCTYPE_REAL * arrayZoneMax,
                             np.zeros(arrayZoneMax, ))
        self.ads.writeByName('.mAO_Heater_PowerLimit', self.ads.PLCTYPE_REAL * arrayZoneMax,
                             np.zeros(arrayZoneMax, ))
        self.ads.writeByName('.mDO_Heater_Command', self.ads.PLCTYPE_BYTE, 1)

    def get_step_time(self):
        return self.step_time

    def get_step_name(self):
        return self.step_name

    def run(self):

        self.connect_plc()

        while True:
            time.sleep(0.5)

            if not self.ads.isConnect:
                self.processing = False
                continue

            ftc_usable = self.ads.readByName('.mFTC_Usable', self.ads.PLCTYPE_BOOL)
            processing_state = self.check_processing()
            ready_state = self.check_ready()

            self.step_time = self.ads.readByName('.mAX_INF_Process_CurrentStepTime', self.ads.PLCTYPE_REAL)
            self.rcp_name = self.ads.readByName('.mSX_INF_Process_CurrentRecipeName', self.ads.PLCTYPE_STRING)
            self.step_name = self.ads.readByName('.mSX_INF_Process_CurrentStepName', self.ads.PLCTYPE_STRING)

            '''
            if self.preStepTime != self.step_time or self.preStepName != self.step_name:
                self.step_log.write('>> StepTime : {} StepName : {}'.format(self.step_time, self.step_name))
                self.preStepTime = self.step_time
                self.preStepName = self.step_name
            '''

            self.control_tc = self.ads.readByName('.mAI_Heater_PV', self.ads.PLCTYPE_REAL * controlTcMax)
            self.glass_tc = self.ads.readByName('.mAI_Recorder_TC_Data', self.ads.PLCTYPE_REAL * glassTcMax)
            self.heater_sp = self.ads.readByName('.mAO_Heater_SP', self.ads.PLCTYPE_REAL * arrayZoneMax)

            if not ftc_usable or processing_state != EnumProcessingState.PROCESSING or ready_state != EnumReadyState.OK:
                self.processing = False
                continue

            self.processing = True
