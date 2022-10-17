# !/usr/bin/env python
import os
import torch
import torch.nn as nn
import rospy
import time
import numpy as np
import pandas as pd
# import pytorch_lightning as pl
# from pytorch_forecasting.data.examples import generate_ar_data
# from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import copy
import copy as copy_module
from copy import deepcopy
from geometry_msgs.msg import WrenchStamped, Pose, PoseStamped, Vector3, Quaternion
from std_msgs.msg import Bool
# from std_msgs.msg import Int8
force_intrxn = []
torque_intrxn = []
force_intrxn.append(0)  # didn't need this before but need it with newer python maybe?
force_intrxn.append(0)
force_intrxn.append(0)

torque_intrxn.append(0)
torque_intrxn.append(0)
torque_intrxn.append(0)
interaction_wrench = WrenchStamped()
interaction_start = 0


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        pred = torch.sigmoid(out)
        return pred

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

def callback_interaction_sensor(msg):
    global force_intrxn, torque_intrxn, interaction_start
    # global interaction_wrench   ## AVOID THIS!, better give forces and torques in this from the code
    temp_force = []
    temp_torque = []
    temp_torque.append(msg.wrench.torque.x)
    temp_force.append(msg.wrench.force.x)
    temp_torque.append(msg.wrench.torque.y)
    temp_force.append(msg.wrench.force.y)
    temp_torque.append(msg.wrench.torque.z)
    temp_force.append(msg.wrench.force.z)
    # print(msg.wrench.force.x)
    # interaction_wrench=msg
    interaction_start = 1
    force_intrxn = deepcopy(temp_force)
    torque_intrxn = deepcopy(temp_torque)


def main():
    global interaction_start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    frequency = 120
    rospy.init_node('Classifier_handover_ML_based', anonymous=False)
    rate = rospy.Rate(frequency)  # 10hz
    time.sleep(2)
    rospy.Subscriber("/interaction_wrench_repub", WrenchStamped, callback_interaction_sensor)
    # pub = rospy.Publisher('/interaction_wrench_repub2', WrenchStamped, queue_size=100)
    pub_pk = rospy.Publisher('/isHandover_Bool', Bool, queue_size=10)
    # zero_int_force_y = force_intrxn[1]  ######################NEED to do this for mostly all the forces! .. done in the repub publisher itself!
    
    keep_going=True
    learning_rate = 0.0001
    num_LSTM_layers = 2
    # bool Ishandover=False
    num_hidden_units = 40
    buffer_Interaction_featrues_input = []


    # ONE WAY IS TO PUT ABOVE AS 200 , THIS GIVES OUT OF SAME DIM AS Y=200

   
    new_model = LSTMClassifier(input_dim=6, hidden_dim=num_hidden_units, layer_dim=num_LSTM_layers,
                               output_dim=1)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
    
    PATH = '/home/parag/pk_model_LSTM_Classifier_370_new_mix'

    new_model.load_state_dict(torch.load(PATH))
    new_model.to(device)
    print(new_model.parameters())
    new_model.eval()
    is_handover = 0
    interaction_flag = 0
    counter = 0

    try:
        # print_commands()
        while not (rospy.is_shutdown() and keep_going):

            time_start = time.time()

            features = [force_intrxn[0], force_intrxn[1], force_intrxn[2], torque_intrxn[0], torque_intrxn[1], torque_intrxn[2]]
            buffer_Interaction_featrues_input.append(features)
            # keep the buffer size <= 100
            if len(buffer_Interaction_featrues_input) == 101:
                buffer_Interaction_featrues_input.pop(0)      # pop , remover literally the first element
            # print(len(buffer_Interaction_featrues_input))
            if interaction_start == 1:
                interaction_flag = 1
            if len(buffer_Interaction_featrues_input) == 100 and interaction_start == 1:
                counter=counter+1
                buffer_features_array = np.array(buffer_Interaction_featrues_input)
                buffer_features_tensor = torch.Tensor(buffer_features_array)
                buffer_features_tensor = buffer_features_tensor.transpose(1, 0)
                buffer_features_tensor = buffer_features_tensor.unsqueeze(-1)
                buffer_features_tensor = buffer_features_tensor.transpose(2, 0)
                with torch.no_grad():
                    y_out = new_model(buffer_features_tensor.to(device))
                # print(y_out)
                # y_out.shape

                y_pk = y_out.squeeze()
                # print(y_pk)
                # y_pk.shape
                is_handover = np.round(y_pk.cpu().numpy())
                # print(is_handover)

                # pub.publish(interaction_wrench)
                if counter > 50: # should change to less! causes delay
                    pub_pk.publish(bool(is_handover))
                    print('publishing 1 or 0')
                # interaction_start = 0 # is is needed ? why?
                print("Interaction Yes")

                if counter > 2000:
                    break
             


            else:
                print("Waiting still")
                is_handover = 0
                pub_pk.publish(bool(is_handover))
            time_spent=time.time()-time_start
            # print(time_spent*1000)
            rate.sleep()

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    # rospy.spin()
    main()
