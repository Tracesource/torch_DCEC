import utils
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
import random
import numpy as np



# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params):

    # Note the time
    since = time.time()

    # Unpack parameters   将外部传入的参数赋值
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']

    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():  #model.children()是迭代器，仅会遍历当前层。
                    if hasattr(layer, 'reset_parameters'):   #判断对象是否包含对应属性
                        layer.reset_parameters()      
        model = pretrained_model
    else:
        print("pretrained:",pretrained)
        try:
            model.load_state_dict(torch.load('nets/' + str(pretrained)))
            # model.load_state_dict(torch.load(pretrained))
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
    # for seed in range(0,30):
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    kmeans(model, copy.deepcopy(dl), params)
    utils.print_both(txt_file, 'seed: {}'.format(seed))

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model 定义两个参数存储最好的结果
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution  初始化目标分布
    utils.print_both(txt_file, '\nUpdating target distribution')
    output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params)
    target_distribution = target(output_distribution)
    nmi = utils.metrics.nmi(labels, preds_prev)
    ari = utils.metrics.ari(labels, preds_prev)
    acc = utils.metrics.acc(labels, preds_prev)
    utils.print_both(txt_file,
                    'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

    #寻找seed
    # return model
    
    if board:
        niter = 0
        writer.add_scalar('/NMI', nmi, niter)
        writer.add_scalar('/ARI', ari, niter)
        writer.add_scalar('/Acc', acc, niter)

    update_iter = 1
    finished = False

    # Go through all epochs
    for epoch in range(num_epochs):

        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)

        schedulers[0].step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data

            inputs = inputs.to(device)

            # Uptade target distribution, chack and print performance 只在指定间隔步更新
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                utils.print_both(txt_file, '\nUpdating target distribution:')
                output_distribution, labels, preds = calculate_predictions(model, dataloader, params)
                target_distribution = target(output_distribution)
                nmi = utils.metrics.nmi(labels, preds)
                ari = utils.metrics.ari(labels, preds)
                acc = utils.metrics.acc(labels, preds)
                utils.print_both(txt_file,
                                 'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                if board:
                    niter = update_iter
                    writer.add_scalar('/NMI', nmi, niter)
                    writer.add_scalar('/ARI', ari, niter)
                    writer.add_scalar('/Acc', acc, niter)
                    update_iter += 1

                # check stop criterion 更新完分布并得到聚类结果后检查迭代是否停止
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _ = model(inputs)
                size_inputs = inputs.shape
                size_outputs = outputs.shape
                with torch.no_grad():
                    a = inputs.clone()
                    a.resize_(size_inputs[0],size_inputs[1],size_inputs[2]*size_inputs[3])  
                    b = outputs.clone()
                    b.resize_(size_outputs[0],size_outputs[1],size_outputs[2]*size_outputs[3])
                # print("a.shape",a.shape)
                # print("b.shape",b.shape)
                # loss_rec = criteria[1](a, b)    #计算重构误差
                # loss_rec = torch.sum(loss_rec) / batch
                loss_rec = criteria[0](inputs, outputs) 
                # loss_rec = criteria[0](inputs, outputs)
                loss_clust = gamma *criteria[2](torch.log(clusters), tar_dist) / batch   #计算聚类损失
                loss = loss_rec + loss_clust
                # print("loss_rec=",loss_rec,"loss_clust=",loss_clust,"loss=",loss)
                loss.backward()
                optimizers[0].step()
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                            epoch_loss_rec,
                                                                                                            epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights 把结果最好的参数保存下来
    model.load_state_dict(best_model_wts)
    return model


# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs,_= data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _ = model(inputs)
                loss = criterion(inputs, outputs)
                loss.backward()
                optimizer.step()

            #  
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)

    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(params['device'])
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 50000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))  #在进行kmeans之后马上更新参数
    # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])
        _, outputs, _ = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    # print(output_array.shape)
    return output_array, label_array, preds


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist
