import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import numpy as np
from main_model import MainModel
from dataloader import load_data_ns, load_data_sevir


def simple_logger(message):
        print(message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphODE_Fourier')
    parser.add_argument('--data_filepath', type=str, default='/data/')
    parser.add_argument("--epochs", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default='sevir', help="dataset")  
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')


    args = parser.parse_args()
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fourier2d_config = {
        'modes1': 12,
        'modes2': 12,
        'pred_len': args.seq_len,
        'width': 20,
    }

    sgode_config = {
        'input_dim': 1, 
        'seq_len': args.seq_len, 
        'horizon': args.seq_len,  
        'num_nodes': 4096,  
        'rnn_units': 64, 
        'embed_dim': 10,
        'Atype': 2,  
        'max_diffusion_step': 2,  
        'cl_decay_steps': 1000, 
        'use_ode_for_gru': True, 
        'filter_type': 'laplacian',
        'logger': simple_logger,
        'temperature': 1.0,
    }

    # dataset = NS2DDataset(args.data_filepath)
    # train_loader, val_loader, test_loader = load_data_ns(batch_size=args.batch_size, val_batch_size=args.val_batch_size, dataset=dataset, num_workers=0)
    # train_loader, val_loader, test_loader, mean, std = load_data_ns(batch_size=args.batch_size, val_batch_size=args.val_batch_size, data_root=args.data_filepath, num_workers=8)
    train_loader, val_loader, test_loader, mean, std = load_data_sevir(batch_size=args.batch_size, val_batch_size=args.val_batch_size, data_root=args.data_filepath, num_workers=8)

    model = MainModel(fourier2d_config, sgode_config, device=device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(time.asctime(time.localtime(time.time())))

    ##################################### Train Val Test #####################################
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            targets = targets.to(device)
            # print(data.shape)
            # print(targets.shape)

            preds = model(data, targets)  # torch.Size([5, 64, 64, 10])
            # print(preds.shape)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Loss: {loss.item()}")

        print(f"Epoch {epoch+1} finished, Avg Loss: {total_loss / (batch_idx+1)}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (data, targets) in enumerate(val_loader):
                data = data.to(device)
                targets = targets.to(device)
                preds = model(data, targets)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        eval_loss = val_loss / len(val_loader)
        
        if eval_loss < best_loss:
            best_loss = eval_loss
            print(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), 'best_model_weights_' + args.dataset + '.pth')


    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device)
            targets = targets.to(device)
            preds = model(data, targets)
            loss = criterion(preds, targets)
            val_loss += loss.item()
        print(f"Validation Avg Loss: {val_loss / len(val_loader)}")
         

    all_labels = []
    all_predictions = []

    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            preds = model(data, targets)
            loss = criterion(preds, targets)
            test_loss += loss.item()
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

        print(f"Test Avg Loss: {test_loss / len(test_loader)}")

    all_predictions_array = np.concatenate(all_predictions, axis=0)
    all_labels_array = np.concatenate(all_labels, axis=0)

    np.save('/results/predictions_' + args.dataset + '.npy', all_predictions_array)
    np.save('/results/labels_' + args.dataset + '.npy', all_labels_array)
    print("Predictions and labels saved as .npy files.")
    print(time.asctime(time.localtime(time.time())))