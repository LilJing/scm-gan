import time
import numpy as np
import torch
import imutil

def visualize_bptt(z, transition, reward_predictor, decoder, rgb_decoder, num_actions, vid=None):
    z.retain_grad()
    actions = []
    zees = []
    if vid is None:
        vid = imutil.Video(filename='excitation_bptt_{}.mp4'.format(int(time.time())), framerate=10)
    for t in range(30):
        a = onehot(1) if t == 0 else onehot(3)
        a.requires_grad = True
        a.retain_grad()

        actions.append(a)  # Keep track of previous actions

        z = transition(z, a)
        z.retain_grad()
        zees.append(z)

        r, rmap = reward_predictor(z, visualize=True)
        r.retain_grad()

        caption = 'Neural Simulation: expected r = {:.2f} {:.2f}'.format(r[0, 0], r[0, 1])
        vid.write_frame(rgb_decoder(decoder(z))[0], resize_to=(512, 512), caption=caption)
        rewards = rmap[0].sum(dim=0)
        rewards = torch.clamp(rewards * 128 + 128, 0, 255)
        #imutil.show(rewards, resize_to=(256, 256), normalize=False, save=False)
        if r.sum().abs() > 0.8:
            print('Expected reward of {:.2f} at time t+{}'.format(r.sum(), t))
            for _ in range(20):
                vid.write_frame(rgb_decoder(decoder(z))[0], resize_to=(512, 512), caption=caption)
            localized_expected_reward = (rmap * (rmap.abs() == rmap.abs().max()).type(torch.cuda.FloatTensor)).sum()
            localized_expected_reward.backward(retain_graph=True)
            print([at*at.grad for at in actions])
            '''
            for z in zees[::-1] + zees:
                caption = 'Plan for reward R={:.2f} at time t+{}'.format(r.sum(), t)
                mask = (z.grad.abs() / (.001 + z.grad.abs().max())) ** 0.5
                img = rgb_decoder(decoder(z * mask))[0]
                for _ in range(4):
                    vid.write_frame(img, resize_to=(512,512), img_padding=8, caption=caption)
            '''
            for z in zees[::-1]:
                caption = 'Causal Backtrack, reward R={:.2f} at time t+{}'.format(r.sum(), t)
                mask = (z.grad.abs() / (.001 + z.grad.abs().max()))
                img1 = decoder(z * mask)[0].sum(dim=0)
                for _ in range(4):
                    vid.write_frame(img1, resize_to=(512,512), img_padding=8, caption=caption)
            break
    return True


def onehot(a_idx, num_actions=4):
    if type(a_idx) is int:
        # Usage: onehot(2)
        return torch.eye(num_actions)[a_idx].unsqueeze(0).cuda()
    # Usage: onehot([1,2,3])
    return torch.eye(num_actions)[a_idx].cuda()
