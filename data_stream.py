import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


class RnwStateStream:

    def __init__(self):
        self.info = StreamInfo('rnw_state', 'data', 10, 100, 'float32', 'myuid34234')

        self.channel_names = [
            'x', 'y', 'psi', 'theta', 'phi',
            'x_dot', 'y_dot', 'psi_dot', 'theta_dot', 'phi_dot'
        ]

        # append some meta-data
        self.info.desc().append_child_value("manufacturer", "LearningRNW")
        channels = self.info.desc().append_child("channels")
        for label in self.channel_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
        self.info.desc().append_child_value("manufacturer", "LSLExamples")
        cap = self.info.desc().append_child("cap")
        cap.append_child_value("name", "ComfyCap")
        cap.append_child_value("size", "54")
        cap.append_child_value("labelscheme", "10-20")

        self.outlet = StreamOutlet(self.info)

    def send(self, data):
        self.outlet.push_sample(data)
