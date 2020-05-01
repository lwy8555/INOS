import math
import torch
import numpy as np
import torch.nn as nn

def eval_dataset(model, data_loader, num_classes, device, samples=1, calc_reconstruction=False, autoregression=False):
    """
    Evaluates an entire dataset with the unified model and stores z values, latent mus and sigmas and output
    predictions according to whether the classification is correct or not.
    The values for correct predictions can later be used for plotting or fitting of Weibull models.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        samples (int): Number of variational samples.
        calc_reconstruction (bool): Option to turn on/off calculation of the decoder for all samples as it is very
                                    computationally heavy and the user might only be interested in the latent space.
        autoregression (bool): flag to indicate whether model is decoding in an autoregressive fashion.

    Returns:
        dict: Dictionary of results and latent values, separated by whether the classification was correct or not.
    """

    # switch to evaluation mode
    model.eval()

    correctly_identified = 0
    tot_samples = 0

    recon_loss_mus = []
    recon_loss_sigmas = []
    out_entropy = []
    out_mus_correct = []
    out_sigmas_correct = []
    out_mus_false = []
    out_sigmas_false = []
    encoded_mus_correct = []
    encoded_mus_false = []
    encoded_sigmas_correct = []
    encoded_sigmas_false = []
    zs_correct = []
    zs_false = []

    for i in range(num_classes):
        out_mus_correct.append([])
        out_mus_false.append([])
        out_sigmas_correct.append([])
        out_sigmas_false.append([])
        encoded_mus_correct.append([])
        encoded_mus_false.append([])
        encoded_sigmas_correct.append([])
        encoded_sigmas_false.append([])
        zs_false.append([])
        zs_correct.append([])

    if autoregression:
        recon_loss = nn.CrossEntropyLoss(reduction='sum')
    else:
        recon_loss = nn.BCEWithLogitsLoss(reduction='none')

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)
            encoded_mu, encoded_std = model.module.encode(inputs)

            out_samples = torch.zeros(samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(samples, encoded_mu.size(0), encoded_mu.size(1)).to(device)
            recon_loss_samples = torch.zeros(samples, inputs.size(0)).to(device)

            if autoregression:
                recon_target = (inputs * 255).long()
            else:
                recon_target = inputs

            # sampling z and classifying
            for i in range(samples):
                z = model.module.reparameterize(encoded_mu, encoded_std)
                z_samples[i] = z

                cl = model.module.classifier(z)
                out = torch.nn.functional.softmax(cl, dim=1)
                out_samples[i] = out

                if calc_reconstruction:
                    dec = model.module.decode(z)

                    if autoregression:
                        # autoregressive loss in bits per dimension
                        dec = model.module.pixelcnn(inputs, torch.sigmoid(dec)).contiguous()
                        recon_loss_samples[i] = (recon_loss(dec, recon_target) / torch.numel(inputs)) \
                                                * math.log2(math.e)
                    else:
                        recon_loss_samples[i] = recon_loss(dec, recon_target).sum(dim=[1, 2, 3])

            # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
            out_mean = out_samples.mean(dim=0)
            out_std = out_samples.std(dim=0)
            zs_mean = z_samples.mean(dim=0)

            eps = 1e-10
            out_entropy.append(-torch.sum(out_mean * torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            if calc_reconstruction:
                recon_loss_mu = recon_loss_samples.mean(dim=0)
                recon_loss_sigma = recon_loss_samples.std(dim=0)

            # for each input and respective prediction store independently depending on whether classification was
            # correct. The list of correct classifications is later used for fitting of Weibull models if the
            # data_loader is loading the training set.
            for i in range(inputs.size(0)):
                tot_samples += 1

                if calc_reconstruction:
                    recon_loss_mus.append(recon_loss_mu[i].item())
                    recon_loss_sigmas.append(recon_loss_sigma[i].item())

                idx = torch.argmax(out_mean[i]).item()
                if classes[i].item() != idx:
                    out_mus_false[idx].append(out_mean[i][idx].item())
                    out_sigmas_false[idx].append(out_std[i][idx].item())
                    encoded_mus_false[idx].append(encoded_mu[i].data)
                    encoded_sigmas_false[idx].append(encoded_std[i].data)
                    zs_false[idx].append(zs_mean[i].data)
                else:
                    correctly_identified += 1
                    out_mus_correct[idx].append(out_mean[i][idx].item())
                    out_sigmas_correct[idx].append(out_std[i][idx].item())
                    encoded_mus_correct[idx].append(encoded_mu[i].data)
                    encoded_sigmas_correct[idx].append(encoded_std[i].data)
                    zs_correct[idx].append(zs_mean[i].data)

    acc = correctly_identified / float(tot_samples)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    # stack list of tensors into tensors
    for i in range(len(encoded_mus_correct)):
        if len(encoded_mus_correct[i]) > 0:
            encoded_mus_correct[i] = torch.stack(encoded_mus_correct[i], dim=0)
            encoded_sigmas_correct[i] = torch.stack(encoded_sigmas_correct[i], dim=0)
            zs_correct[i] = torch.stack(zs_correct[i], dim=0)
        if len(encoded_mus_false[i]) > 0:
            encoded_mus_false[i] = torch.stack(encoded_mus_false[i], dim=0)
            encoded_sigmas_false[i] = torch.stack(encoded_sigmas_false[i], dim=0)
            zs_false[i] = torch.stack(zs_false[i], dim=0)

    # Return a dictionary containing all the stored values
    return {"accuracy": acc, "encoded_mus_correct": encoded_mus_correct, "encoded_mus_false": encoded_mus_false,
            "encoded_sigmas_correct": encoded_sigmas_correct, "encoded_sigmas_false": encoded_sigmas_false,
            "zs_correct": zs_correct, "zs_false": zs_false,
            "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
            "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false, "out_entropy": out_entropy,
            "recon_loss_mus": recon_loss_mus, "recon_loss_sigmas": recon_loss_sigmas}
