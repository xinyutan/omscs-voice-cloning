import torch
import torch.nn as nn
from models.hyperparameters import SpeakerVerifierHyperparameters as sv_hp
from models.speaker_verifier import SpeakerVerifier
from data.data_loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def evaluate(model, dev_dataloader):
    
    num_loops = sv_hp.examples_evaluate // sv_hp.batch_size
    probs = []
    labels = []
    for i in range(num_loops):
        X, y = next(dev_dataloader)
        with torch.no_grad():
            model.eval()
            yprob = model.predict_prob(X[:-1], X[-1])
            probs.append(yprob.detach().numpy())
        labels.append(y.detach().numpy())
    
    prob_array = np.ndarray.flatten(np.concatenate(probs, axis=0))
    label_array = np.concatenate(labels, axis=0).astype(bool)

    return roc_auc_score(label_array, prob_array), \
        accuracy_score(prob_array>0.5, label_array)


def train(run_id, train_dataset, dev_dataset, num_epochs, models_dir,
          save_every=0, print_every=0):
    # Setting up the training objects.
    train_dl = DataLoader(train_dataset, sv_hp.batch_size)
    dev_dl = DataLoader(dev_dataset, sv_hp.batch_size)

    model = SpeakerVerifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=sv_hp.learning_rate)

    # Configure the path for the models.
    state_fpath = models_dir.joinpath(run_id + ".pt")

    model.train()
    for step in range(num_epochs):
        # Step.
        optimizer.zero_grad()
        X, y = next(train_dl)
        scores = model.forward(X[:-1], X[-1])
        loss = criterion(scores, y.reshape(-1, 1))

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), sv_hp.gradient_clipping_max_value)
        optimizer.step()

        if print_every != 0 and step % print_every == 0:
            print(f"Loss at step {step}: {loss}")
            print(evaluate(model, dev_dl))       

        if save_every != 0 and step % save_every == 0:
            print(f"Saving the model (step {step}):")
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
        
