from dataset.dataset import SimDataset
import torch
from model import ContactTransformer
import torch.optim as optim
import os
from tqdm import tqdm
import wandb
import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    globals()[cfg.mode](cfg)


def train(cfg):
    # initialize wandb
    wandb.init(
        project=cfg.wandb_project_name,
        reinit=True,
        mode="disabled" if cfg.debug else "online",
        settings=wandb.Settings(start_method="fork"),
        config=OmegaConf.to_container(cfg)
    )
    wandb.run.name = cfg.name
    wandb.run.save()

    dataset = SimDataset(**cfg, split="train", data_path=cfg.train_path)
    testset = SimDataset(**cfg, split="eval", data_path=cfg.eval_path)
    trainLoader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=12,
        )
    testLoader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=True, num_workers=12,
    )
    model = ContactTransformer(**cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse_loss = torch.nn.MSELoss()
    device = torch.device(cfg.device)   
    model.to(device)
    
    best_loss = 100  
    for e in tqdm(range(cfg.num_epoch)):
        running_loss = 0.0
        model.train()
        
        for i, src in enumerate(trainLoader):
            src = {k: v.to(device) for k, v in src.items()}

            optimizer.zero_grad()
            pred_contact = model(**src)
            # (batch, traj_len, 3) and (batch, traj_len, 3)
            loss_contact = mse_loss(pred_contact[:,:,:3], src["position"])
            loss = loss_contact
            loss.backward()
            optimizer.step()
            running_loss = (running_loss * i + loss.item()) / (i + 1)

        wandb.log({
                "train/loss": running_loss,
                "train/mse": loss_contact,
            })
           
        model.eval()
        eval_loss = 0
        for i, src in enumerate(testLoader):
            src = {k: v.to(device) for k, v in src.items()}
            with torch.no_grad():
                pred_contact = model(**src)
                loss_contact = mse_loss(pred_contact[:,:,:3], src["position"])
                eval_loss = (eval_loss * i + loss_contact.item()) / (i + 1)

        wandb.log({
                "eval/loss": eval_loss,
                "eval/mse": loss_contact,
            })
        if eval_loss <= best_loss:
            print("update best model at ", e, "th epoch")
            best_loss = eval_loss
            print("best loss is", best_loss)
            print("saving best model!")
            modelDir = os.path.join("ckpt", cfg.name)
            os.makedirs(modelDir, exist_ok=True)
            bestModel = os.path.join(modelDir, "best.pth")
            torch.save(model.state_dict(), bestModel)
    
        if e%2==0:
            modelDir = os.path.join("ckpt", cfg.name)
            os.makedirs(modelDir, exist_ok=True)
            fnModel = os.path.join(modelDir, "ep" + str(e) + ".pth")
            torch.save(model.state_dict(), fnModel)
    
    modelDir = os.path.join("ckpt", cfg.name)
    os.makedirs(modelDir, exist_ok=True)
    fnModel = os.path.join(modelDir, "final.pth")
    torch.save(model.state_dict(), fnModel)

if __name__ == "__main__":
    main()