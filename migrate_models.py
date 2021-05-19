"""
Migrate saved models to newer versions of pytorch by simply loading and saving
"""
import sys
sys.path.insert(0, './future_image_similarity')
import torch
import argparse


parser = argparse.ArgumentParser()
opt = parser.parse_args([])
predictor = False
# load model
if predictor:
    opt.model_dir = 'future_image_similarity/logs/gaz_pose/model_predictor_pretrained'
    saved_model = torch.load('%s/model.pth' % opt.model_dir)  # https://github.com/pytorch/pytorch/issues/3678
    # loading from old model
    opt = saved_model['opt']
    prior = saved_model['prior']
    encoder = saved_model['encoder']
    decoder = saved_model['decoder']
    pose_network = saved_model['pose_network']
    conv_network = saved_model['conv_network']

    # correcting old model
    # correct opt.data_root
    opt.data_root = "future_image_similarity/data/sim/targets/target1"
    # constructing new models with new pytorch version and loading state_dict from old model.pth
    from future_image_similarity.models.model_predictor import gaussian_lstm as lstm_model
    import future_image_similarity.models.model_predictor as model
    # strict=False https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/6
    pr_new = lstm_model(4*opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    pr_new.load_state_dict(prior.state_dict(), strict=False)  # do load_state_dict in separate line, and don't assign it again (don't pr_new = pr_new.load_...)
    e_new = model.encoder_conv(opt.g_dim, opt.channels)
    e_new.load_state_dict(encoder.state_dict(), strict=False)
    d_new = model.decoder_conv(opt.g_dim, opt.channels)
    d_new.load_state_dict(decoder.state_dict(), strict=False)
    po_new = model.pose_network()
    po_new.load_state_dict(pose_network.state_dict(), strict=False)
    c_new = model.conv_network(16+opt.g_dim+int(opt.z_dim/4), opt.g_dim)
    c_new.load_state_dict(conv_network.state_dict(), strict=False)

    # save model with updated pytorch version
    # torch.save(saved_model, 'logs/gaz_pose/model_predictor_pretrained/model.pth')  # saving new model
    torch.save({
        'pose_network': po_new,
        'conv_network': c_new,
        'encoder': e_new,
        'decoder': d_new,
        'prior': pr_new,
        'opt': opt}, 'logs/gaz_pose/model_predictor_pretrained/model.pth')
    print("done")
    # newly saved model avoids warnings about model saved with old pytorch, corrects data_root
else:
    model_dir = 'future_image_similarity/logs/gaz_value/model_critic_pretrained'
    critic = torch.load('%s/model_critic.pth' % model_dir)
    from future_image_similarity.models.value_network import ModelValue
    critic_new = ModelValue()
    critic_new.load_state_dict(critic.state_dict(), strict=False)
    torch.save({'value_network':critic_new}, 'logs/gaz_value/model_critic_pretrained/model.pth')
    print("done")
    # newly saved model avoids warnings about model saved with old pytorch, corrects data_root
