import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import datetime
from torchsummary import summary
import torch
from dataset import Multimodal_dataset
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from graph_video_audio_model import GAT_video_audio
# from graph_video_model import GAT_video
import random
from sklearn.manifold import TSNE


batch_size = 32
frame_num = 4
continue_train = 0
retrain_epoch = 0
os.environ['CUDA_VISIBLE_DEVICES'] = ''
model_name = ''
load_model_name = ''
epoches = 20
train = 1
test = 1
video_level = 1
tsne = 1
image_size = 128
num_classes = 4
time = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
summary_path = os.path.join('summary/weight/', model_name)
tsne_figure_path = os.path.join('summary/figure/', model_name)
model_path = os.path.join(os.path.join('summary/weight/', load_model_name), '2.pth')




result_train_score = []
result_train_loss = []

result_val_score = []
result_val_loss = []



if __name__ == '__main__':
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(tsne_figure_path):
        os.makedirs(tsne_figure_path)


    train_dataset = Multimodal_dataset(image_size, 'train', 'data_path/fakeav_50_frame_train_path.txt', num_frame=frame_num)
    test_dataset = Multimodal_dataset(image_size, 'test', 'data_path/fakeav_50_frame_test_path.txt', num_frame=frame_num)



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=False,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              drop_last=False,
                                              num_workers=8)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    model = GAT_video_audio(num_classes=num_classes, audio_nodes=4)

    if Device == 'cuda':
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    if continue_train:
        model.load_state_dict(torch.load(model_path))
        print('loading successfully!')

    # weight = torch.from_numpy(np.array(([1.0, 0.2]))).float()
    criterion = nn.CrossEntropyLoss().cuda()
    Bloss = nn.BCELoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = optim.SGD(model.parameters(), lr=0.0003, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    print(time, 'train:', train, 'test:', test, ' ==>', model_name, '\n', image_size, ' begin\n')
    for epoch in range(epoches):
        if train:

            print('Epoch {}/{}'.format(epoch + 1, epoches))
            print('-' * 10)
            model.train()
            train_loss = 0.0
            train_corrects = 0.0
            for (name, video, audio, total_label, video_label, audio_label) in tqdm(train_loader): #(name, video, audio, total_label, video_label, audio_label)
                iter_loss = 0.0
                iter_corrects = 0.0
                video = video.cuda()
                audio = audio.cuda()
                total_label = total_label.cuda()
                video_label = video_label.cuda()
                audio_label = audio_label.cuda()
                optimizer.zero_grad()

                total_output, video_output, audio_output, fusion_output = model(video, audio)

                _, preds = torch.max(total_output.data, 1)
                loss1 = criterion(total_output, total_label)
                loss2 = criterion(video_output, video_label)
                loss3 = criterion(audio_output, audio_label)
                loss = loss1 + loss2 + loss3

                # print(outputs,labels)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                iter_loss = loss.data.item()
                train_loss += iter_loss
                iter_corrects = torch.sum(preds == total_label.data).to(torch.float32)
                train_corrects += iter_corrects
                iteration += 1
                if not (iteration % 5000):
                    print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size,
                                                                               iter_corrects / batch_size))
                    # print(model.module.gate)

            epoch_loss = train_loss / train_dataset_size
            epoch_acc = train_corrects / train_dataset_size
            print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            result_train_loss.append(epoch_loss)
            result_train_score.append(epoch_acc)
            with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                file.write(
                    'epoch: ' + str(epoch+1+retrain_epoch) + ' train loss: ' + str(epoch_loss) + ' train_acc: ' + str(epoch_acc) + '\n')

        roc_pre = []
        roc_lab = []
        confusion_matrix_pre = []
        confusion_matrix_lab = []
        label_dict = {}
        count_dict = {}
        result_fea_list = []
        result_label_list = []
        if test:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_corrects = 0.0
                for (name, video, audio, total_label, video_label, audio_label) in tqdm(test_loader):
                    result_label_list.append(total_label)
                    for i in range(len(name)):
                        label_dict[name[i]] = total_label[i]
                    video = video.cuda()
                    audio = audio.cuda()
                    total_label = total_label.cuda()
                    video_label = video_label.cuda()
                    audio_label = audio_label.cuda()

                    total_output, video_output, audio_output, fusion_output = model(video, audio)

                    _, preds = torch.max(total_output.data, 1)
                    loss1 = criterion(total_output, total_label)
                    loss2 = criterion(video_output, video_label)
                    loss3 = criterion(audio_output, audio_label)
                    # loss4 = Closs(fusion_output, total_label)
                    loss = loss1 + loss2 + loss3

                    result_fea_list.append(fusion_output.cpu().numpy())


                    test_loss += loss.data.item()
                    test_corrects += torch.sum(preds == total_label.data).to(torch.float32)
                    confusion_matrix_pre.append(preds)
                    confusion_matrix_lab.append(total_label)

                    softmax_out = torch.softmax(total_output, dim=-1)
                    roc_pre.append(softmax_out)
                    # label_roc = total_label.view(-1, 1)
                    # label_roc = torch.cat((1 - label_roc, label_roc), dim=1)
                    # roc_lab.append(label_roc)
                    roc_lab.append(F.one_hot(total_label, num_classes=num_classes))       #4


                    #预测分数求和
                    predstonp = softmax_out.cpu().numpy()
                    for i in range(len(name)):
                        if name[i] not in count_dict:
                            count_dict[name[i]] = torch.zeros(num_classes)       #4
                            count_dict[name[i]] += predstonp[i]
                        else:
                            count_dict[name[i]] += predstonp[i]


                epoch_loss = test_loss / test_dataset_size
                epoch_acc = test_corrects / test_dataset_size

                epoch_auc = roc_auc_score(torch.cat(roc_lab, dim=0).cpu().data, torch.cat(roc_pre, dim=0).cpu().data)
                epoch_confuison_matrix = confusion_matrix(torch.cat(confusion_matrix_lab, dim=0).cpu().data,
                                                          torch.cat(confusion_matrix_pre, dim=0).cpu().data)

                print('epoch test loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, epoch_auc))
                print('confuison_matrix:')
                print(epoch_confuison_matrix)
                result_val_loss.append(epoch_loss)
                result_val_score.append(epoch_acc)
                if epoch_acc >= best_acc:

                    if video_level:
                        pre = []
                        lab = []
                        length = len(label_dict)
                        for value in label_dict.values():
                            temp = value.unsqueeze(0)
                            lab.append(temp)

                        for value in count_dict.values():
                            temp = value.unsqueeze(0)
                            pre.append(temp)

                        lab = torch.cat(lab, dim=0)
                        pre = torch.cat(pre, dim=0)
                        _, index = torch.max(pre, 1)
                        iter_corrects = torch.sum(index == lab).to(torch.float32)
                        print('video_level_acc:', iter_corrects / length)

                        # lab_roc = lab.view(-1, 1)
                        # lab_roc = torch.cat((1 - lab_roc, lab_roc), dim=1)
                        lab_roc = F.one_hot(lab, num_classes=num_classes)                #4
                        pre_roc = torch.softmax(pre, dim=-1)
                        # # pre_roc = pre
                        video_auc = roc_auc_score(lab_roc.cpu().data, pre_roc.cpu().data)
                        video_confusion_matrix = confusion_matrix(lab.cpu().data, index.cpu().data)
                        print('video_level_auc:', video_auc)
                        print('video_confusion_matrix:')
                        print(video_confusion_matrix)

                        with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                            file.write(
                                'epoch:' + str(epoch + 1 + retrain_epoch) + ' video_level_acc: ' + str(iter_corrects / length) +
                                ' video_level_auc: ' + str(video_auc)+ '\n' +'video_level_confusion_matrix: '+ str(video_confusion_matrix)+'\n')

                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, os.path.join(summary_path, str(epoch+1+retrain_epoch) + '.pth'))
                with open('summary/result/' + model_name + '_' + time + '.txt', 'a+') as file:
                    file.write('epoch:' + str(epoch+1+retrain_epoch) + ' test loss: ' + str(epoch_loss) + ' test_acc: ' + str(
                        epoch_acc) + ' test auc: ' + str(epoch_auc)+ '\n' + 'frame_confusion_matrix: '+ str(epoch_confuison_matrix) + '\n')

                scheduler.step()

            print('Best test Acc: {:.4f}'.format(best_acc))
            print('\n')




