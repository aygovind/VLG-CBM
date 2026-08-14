import os
import argparse
import datetime
import json
import torch
from model.cbm import train_dense_final, train_sparse_final

import clip
import model.utils as utils
from data import utils as data_utils

from glm_saga.elasticnet import IndexedTensorDataset
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Settings for creating model')

parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--lam", type=float, default=0.0125, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--dense", action="store_true", help="train with dense or sparse method")
parser.add_argument("--dense_lr", type=float, default=0.001, help="Learning rate for the dense final layer training")

def train_and_save(args):
    #load data and models
    
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"
    
    target_model, target_preprocess = data_utils.get_target_model(args.backbone, args.device)

    data_t = data_utils.get_data(d_train, preprocess=target_preprocess)
    val_data_t = data_utils.get_data(d_val, preprocess=target_preprocess)
    
    with open(data_utils.LABEL_FILES[args.dataset], "r") as f:
        classes = f.read().split("\n")
    
    target_save_name, _, _ = utils.get_save_names("", args.backbone, args.feature_layer, d_train, "", "avg",
                                                  args.activation_dir)
    val_target_save_name, _, _ =  utils.get_save_names("", args.backbone, args.feature_layer, d_val, "", "avg",
                                                       args.activation_dir)
    #save activations and get save_paths
    if args.backbone.startswith("clip_"):
        model, _ = clip.load(args.backbone[5:], device=args.device)
        utils.save_clip_image_features(model, data_t, target_save_name, args.batch_size, args.device)
        utils.save_clip_image_features(model, val_data_t, val_target_save_name, args.batch_size, args.device)
    else:
        utils.save_target_activations(target_model, data_t, target_save_name, target_layers = [args.feature_layer],
                                  batch_size = args.batch_size, device = args.device, pool_mode='avg')
        utils.save_target_activations(target_model, val_data_t, val_target_save_name, target_layers = [args.feature_layer],
                                  batch_size = args.batch_size, device = args.device, pool_mode='avg')
    

    #load features
    target_features = torch.load(target_save_name, map_location="cpu").float()
    val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
    
    with torch.no_grad():
        train_c = target_features.detach()
        val_c = val_target_features.detach()
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)

        # Keep the features resident on the GPU. Left on the CPU, every batch of every
        # SAGA iteration is copied across the bus and the solver stalls on a
        # single-threaded loader -- measured at 12% GPU utilisation in the LF-CBM repo,
        # which is both slow and below the NRP 40% floor. SAGA is this script's entire
        # workload, so that is the whole run. birds525 is 260MB at 768-d, 693MB for
        # resnet50's 2048-d.
        # Features move to the GPU; labels must NOT. glm_saga builds its one-hot targets
        # with `I[y]` against a CPU identity matrix (elasticnet.py:321), so CUDA label
        # indices raise "indices should be either on cpu or on the same device".
        train_z = ((train_c-train_mean)/train_std).to(args.device)
        labels = data_t.targets
        train_y = torch.LongTensor(labels)

        indexed_train_ds = IndexedTensorDataset(train_z,train_y)

        val_z = ((val_c-train_mean)/train_std).to(args.device)
        val_labels = val_data_t.targets
        val_y = torch.LongTensor(val_labels)

        val_ds = TensorDataset(val_z,val_y)


    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    model = torch.nn.Linear(train_z.shape[1], len(classes)).to(args.device)
    if args.dense:
        output_proj = train_dense_final(model, indexed_train_loader, val_loader, args.n_iters, args.dense_lr, device=args.device)
    else:
        output_proj = train_sparse_final(model, indexed_train_loader, val_loader, args.n_iters, args.lam, device=args.device)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    W_g = W_g.to(args.device)
    
    save_name = "{}/{}_finetuned_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    # makedirs, not mkdir: --save_dir is a per-backbone directory that does not exist on
    # the first run, and mkdir only creates the leaf. This fires after training, so the
    # crash would land at the end of the run and throw the work away.
    os.makedirs(save_name, exist_ok=True)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_and_save(args)