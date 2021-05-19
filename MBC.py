
import torch
import sys
sys.path.insert(0, './future_image_similarity')
model_dir = 'future_image_similarity/logs/gaz_value/model_critic_pretrained'
saved_model = torch.load('%s/model_critic.pth' % model_dir)
print()
class MBC_Agent():
    pass

