
import torch
import sys
sys.path.insert(0, './future_image_similarity')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MBC_Agent():
    def __init__(self, env):
        """ env: DreamGazeboEnv"""
        model_dir = 'logs/gaz_value/model_critic_pretrained'
        self.critic = torch.load('%s/model.pth' % model_dir)['value_network']
        self.critic.to(device)
        self.env = env
    
    def select_action(self, state, candidates):
        value = []
        for candidate in candidates:
            x_pred = self.env.predict(state, candidate)
            x_value = self.critic(x_pred)
            value.append(x_value)

        value = torch.stack(value)
        best_pose_idx = torch.argmax(-value)
        delta_pose = candidates[best_pose_idx]
        return delta_pose

