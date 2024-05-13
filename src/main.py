# Import libraries
import torch, wandb, argparse, yaml, os, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from dataset import get_dls
from transformations import get_tfs
from time import time
from train import train_setup, train
from utils import DrawLearningCurves
from pl_train import CustomModel, ImagePredictionLogger

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    pretrained_params_path = f"{args.pretrained_params_path}/{args.data_name}"
    
    os.makedirs(f"{args.dls_dir}/{args.data_name}", exist_ok = True); os.makedirs(f"{args.stats_dir}/{args.data_name}", exist_ok = True); 
    os.makedirs(pretrained_params_path, exist_ok = True)
    os.system(f"wandb login --relogin {args.wandb_key}")
    inp_im_size = (224, 224) if args.data_name == "flowers" else ((28, 28) if args.data_name == "mnist" else (32, 32))
    transformations = get_tfs(im_size = inp_im_size, gray = True) if args.data_name == "mnist" else get_tfs(im_size = inp_im_size)
    class_file = f"{args.dls_dir}/{args.data_name}/cls_names"
    tr_dl_file = f"{args.dls_dir}/{args.data_name}/tr_dl_gpu_{args.devices}_devices"
    vl_dl_file = f"{args.dls_dir}/{args.data_name}/vl_dl_gpu_{args.devices}_devices"
    ts_dl_file = f"{args.dls_dir}/{args.data_name}/ts_dl_gpu_{args.devices}_devices"
    
    if os.path.isfile(tr_dl_file) and os.path.isfile(vl_dl_file) and os.path.isfile(ts_dl_file) and os.path.isfile(class_file): 
        tr_dl, val_dl, ts_dl, classes = torch.load(tr_dl_file), torch.load(vl_dl_file), torch.load(ts_dl_file), torch.load(class_file)
    else:
        tr_dl, val_dl, ts_dl, classes = get_dls(root = args.root, transformations = transformations, bs = args.batch_size, ds_name = args.data_name)
        torch.save(tr_dl,   tr_dl_file); torch.save(val_dl,  vl_dl_file); torch.save(ts_dl, ts_dl_file); torch.save(obj = classes, f = class_file)
    
    in_fs = 64 if args.data_name == "mnist" else 256
    if args.train_framework == "pl":
        
        for act_name in ["regular", "wib", "leaky", "prelu", "gelu"]:

            ckpt_name = f"best_model_{act_name}_{args.train_framework}"
            # Samples required by the custom ImagePredictionLogger callback to log image predictions. 
            val_samples = next(iter(val_dl))

            model = CustomModel(input_shape = inp_im_size, act_name = act_name, in_fs = in_fs, num_classes = len(classes), lr = args.learning_rate) 

            # Initialize wandb logger
            wandb_logger = WandbLogger(project = f"{args.data_name}", job_type = "train", name = f"{act_name}_{args.data_name}_{args.batch_size}_{args.learning_rate}")

            # Initialize a trainer
            trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", 
                                 logger = wandb_logger, 
                                 # fast_dev_run = True,
                                 callbacks = [EarlyStopping(monitor = "val_loss", mode = "min", patience = 5), ImagePredictionLogger(val_samples, classes),
                                              ModelCheckpoint(monitor = "val_loss", dirpath = pretrained_params_path, filename = ckpt_name)])

            # Train the model
            trainer.fit(model, tr_dl, val_dl)

            # Test the model
            trainer.test(ckpt_path = f"{pretrained_params_path}/{ckpt_name}.ckpt", dataloaders = ts_dl)

            # Close wandb run
            wandb.finish()

    elif args.train_framework == "py":

        m, epochs, device, loss_fn, optimizer = train_setup(act_name = act_name, epochs = args.epochs, classes = classes, device = args.device)
        results = train(tr_dl = tr_dl, val_dl = val_dl, m = m, device = args.device, 
                        loss_fn = loss_fn, optimizer = optimizer, epochs = args.epochs, 
                        save_dir = pretrained_params_path, save_prefix = args.data_name, train_framework = args.train_framework)

        DrawLearningCurves(results, f"{args.stats_dir}/{args.data_name}").save_learning_curves()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "data", help = "Path to data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 128, help = "Mini-batch size")
    parser.add_argument("-dn", "--data_name", type = str, default = "cifar10", help = "Dataset name for the experiments")
    parser.add_argument("-ds", "--devices", type = int, default = 4, help = "GPU devices number")
    parser.add_argument("-d", "--device", type = str, default = "cuda:2", help = "GPU device name")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 3e-4, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "Train epochs number")
    parser.add_argument("-sm", "--pretrained_params_path", type = str, default = 'pretrained_params', help = "Path to the directory to save pretrained parameters")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    parser.add_argument("-wk", "--wandb_key", type = str, default = "3204eaa1400fed115e40f43c7c6a5d62a0867ed1", help = "Wandb key can be obtained from wandb.ai")
    parser.add_argument("-tf", "--train_framework", type = str, default = "pl", help = "Framework to be used for training an AI model")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)