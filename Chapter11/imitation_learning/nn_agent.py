#!/usr/bin/env python
from __future__ import print_function

import os

import gym
import numpy as np
import torch
from train \
    import \
    data_transform, \
    available_actions, \
    build_network, \
    DATA_DIR, MODEL_FILE


def nn_agent_drive(model: torch.nn.Module, device: torch.device):
    """
    Let the agent drive
    :param model: the network
    :param device: the cuda device
    """

    env = gym.make('CarRacing-v0')

    global human_wants_exit  # use ESC to exit
    human_wants_exit = False

    def key_press(key, mod):
        """Capture ESC key"""
        global human_wants_exit
        if key == 0xff1b:  # escape
            human_wants_exit = True

    state = env.reset()  # initialize environment
    env.unwrapped.viewer.window.on_key_press = key_press

    while 1:
        env.render()

        state = np.moveaxis(state, 2, 0)  # channel first image
        state = torch.from_numpy(np.flip(state, axis=0).copy())  # np to tensor
        state = data_transform(state).unsqueeze(0)  # apply transformations
        state = state.to(device)  # add additional dimension

        with torch.set_grad_enabled(False):  # forward
            outputs = model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)

        # translate from net output to env action
        max_action = np.argmax(normalized.cpu().numpy()[0])
        action = available_actions[max_action]
        action[2] = 0.3 if action[2] != 0 else 0  # adjust brake power

        state, _, terminal, _ = env.step(action)  # one step

        if terminal:
            state = env.reset()

        if human_wants_exit:
            env.close()
            return


if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = build_network()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    m.eval()
    m = m.to(dev)
    nn_agent_drive(m, dev)
