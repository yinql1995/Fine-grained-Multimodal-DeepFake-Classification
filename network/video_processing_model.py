import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from torchsummary import summary

num_frames = 4


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*num_frames, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)

    def forward(self, x):
        fuse_fea = self.convblk(x)
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea


class Horizontal_Vertical_att(nn.Module):
    def __init__(self, in_filters, ratio=4):
        super(Horizontal_Vertical_att, self).__init__()

        self.horizontal_conv = nn.Sequential(
            SeparableConv2d(in_filters, in_filters//ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters//ratio),
        )
        self.vertical_conv = nn.Sequential(
            SeparableConv2d(in_filters, in_filters//ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters//ratio),
        )

        self.avg_pool_row = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.avg_pool_row2 = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.avg_pool_col = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.avg_pool_col2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))


        self.horizontal_conv_down = nn.Sequential(
            SeparableConv2d(in_filters, in_filters//ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters//ratio),
        )
        self.vertical_conv_down = nn.Sequential(
            SeparableConv2d(in_filters, in_filters//ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters//ratio),
        )

        self.horizontal_conv_down2 = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(in_filters//ratio),
        )
        self.vertical_conv_down2 = nn.Sequential(
            SeparableConv2d(in_filters, in_filters // ratio, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(in_filters//ratio),
        )

        self.horizontal_conv_merge = nn.Sequential(
            nn.Conv2d(in_filters//ratio, in_filters, 1),
            nn.BatchNorm2d(in_filters),
        )
        self.vertical_conv_merge = nn.Sequential(
            nn.Conv2d(in_filters//ratio, in_filters, 1),
            nn.BatchNorm2d(in_filters),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        input = x.view((x.size(0)//num_frames, num_frames,)+x.size()[1:])
        B, T, C, H, W = input.size()

        input_col = input.permute(0, 4, 2, 3, 1).contiguous()   # B W C H T
        input_col = input_col.view((-1,)+input_col.size()[2:])
        input_col_down = self.avg_pool_col(input_col)
        input_col_down2= self.avg_pool_col2(input_col)


        input_row = input.permute(0, 3, 2, 1, 4).contiguous()   #B H C T W
        input_row = input_row.view((-1,)+input_row.size()[2:])
        input_row_down = self.avg_pool_row(input_row)
        input_row_down2 = self.avg_pool_row2(input_row)

        input_col = self.vertical_conv(input_col)
        input_row = self.horizontal_conv(input_row)

        input_col_down = self.vertical_conv_down(input_col_down)
        input_row_down = self.horizontal_conv_down(input_row_down)

        input_col_down2 = self.vertical_conv_down2(input_col_down2)
        input_row_down2 = self.horizontal_conv_down2(input_row_down2)

        input_col_down = F.interpolate(input_col_down, input_col.size()[2:])
        input_row_down = F.interpolate(input_row_down, input_row.size()[2:])

        input_col_down2 = F.interpolate(input_col_down2, input_col.size()[2:])
        input_row_down2 = F.interpolate(input_row_down2, input_row.size()[2:])

        col = self.vertical_conv_merge(1/3*input_col + 1/3*input_col_down + 1/3*input_col_down2)
        row = self.horizontal_conv_merge(1/3*input_row + 1/3*input_row_down + 1/3*input_row_down2)

        col = self.sigmoid(col)-0.5
        row = self.sigmoid(row)-0.5

        col = col.view((col.size(0)//W, W,)+col.size()[1:]).permute(0, 4, 2, 3, 1).contiguous()
        row = row.view((row.size(0)//H, H,)+row.size()[1:]).permute(0, 3, 2, 1, 4).contiguous()

        att = col*0.5 + row*0.5
        att = att.view((-1,)+att.size()[2:])

        return x + self.gamma * att * x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=False, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class Block_1(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=False, grow_first=True, relu=True, sim=False, hv=False):
        super(Block_1, self).__init__()

        self.neg_sim = spatial_self_attention(in_filters)
        # self.neg_sil  = mSEModule(in_filters)

        self.block = Block(in_filters, out_filters, reps, strides, start_with_relu, grow_first)

        self.relu = nn.ReLU(inplace=True)
        self.flag = relu

        self.hv_att = Horizontal_Vertical_att(in_filters)
        self.sim = sim
        self.hv = hv

    def forward(self, x):

        if self.flag:
            x = self.relu(x)
        if self.sim:
            x = self.neg_sim(x)
        if self.hv:
            x = self.hv_att(x)
        x = self.block(x)

        return x


class spatial_self_attention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(spatial_self_attention, self).__init__()

        self.q_conv = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): #BT, C, H, W

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = q.view((-1, num_frames,) + q.size()[1:])   # B, T, C, H, W
        B, T, C, H, W = q.size()
        q = q.view(B, T, C, -1) .permute(0, 3, 1, 2).contiguous()    # B, HW, T, C
        q = F.normalize(q, dim=-1)

        k = k.view((-1, num_frames,) + k.size()[1:])
        k = k.view(B, T, C, -1).permute(0, 3, 1, 2).contiguous()  # B, HW, T, C
        k = F.normalize(k, dim=-1)

        att = -torch.matmul(q, k.permute(0, 1, 3, 2))  # B, HW, T, T
        att = att.sum(-1)  # B, HW, T

        # mean = torch.mean(att, dim=2, keepdim=True)
        # att = self.relu(att-mean)



        att_var = torch.var(att, dim=2)   # B, HW
        att_var_k = torch.topk(att_var, k=H, largest=True, sorted=True)[0]
        att_var_k = att_var_k[:, -1].unsqueeze(-1)   # B, 1
        att_mask = torch.le(att_var, att_var_k)     #B, HW
        att_mask = att_mask.expand(T, att_mask.size(0), att_mask.size(1)).permute(1, 2, 0).contiguous()
        att[att_mask] = -float(num_frames)  #B, HW, T

        # att = nn.Softmax(dim=1)(att)

        temp_min = torch.min(att, dim=1, keepdim=True)[0]
        temp_max = torch.max(att, dim=1, keepdim=True)[0]
        att = (att - temp_min)/(temp_max-temp_min + 1e-8)


        att = att.permute(0, 2, 1).contiguous().unsqueeze(2)  # B, T, 1, HW

        att = att.view(B, T, 1, H, W)

        att = self.gamma * v * att.view(-1, 1, H, W)  # BT, C H, W

        x = x + att

        return x


class video_model(nn.Module):

    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(video_model, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # self.f_d = mSEModule(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)


        self.block1 = Block_1(64, 64, 2, 2, start_with_relu=False, grow_first=True, relu=False, sim=True, hv=True)

        self.block2 = Block_1(64, 128, 2, 2, start_with_relu=False, grow_first=True, sim=True, hv=True)

        self.block3 = Block_1(128, 128, 2, 2, start_with_relu=False, grow_first=True, sim=True, hv=True)

        # self.block4 = Block_1(512, 512, 3, 1, start_with_relu=False, grow_first=True, sim=True, hv=True)
        # self.block5 = Block_1(512, 512, 3, 1, start_with_relu=False, grow_first=True, sim=True, hv=True)
        # self.block6 = Block_1(512, 512, 3, 1, start_with_relu=False, grow_first=True, sim=True, hv=True)
        # self.block7 = Block_1(512, 512, 3, 2, start_with_relu=False, grow_first=True, sim=True, hv=True)

        # self.block8 = Block_1(728, 728, 3, 1, start_with_relu=False, grow_first=True)
        # self.block9 = Block_1(728, 728, 3, 1, start_with_relu=False, grow_first=True)
        # self.block10 = Block_1(728, 728, 3, 1, start_with_relu=False, grow_first=True)
        # self.block11 = Block_1(728, 728, 3, 1, start_with_relu=False, grow_first=True)

        # self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        # do relu here
        self.conv4 = SeparableConv2d(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fea_fusion = FeatureFusionModule(128*num_frames, 512)

        # self.fc = nn.Linear(512, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):

        x = x.view((-1, int(x.size()[1]//num_frames),) + x.size()[2:])

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)


        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)

        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        # x = self.block11(x)

        # x = self.block12(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = x.view(x.size(0) // num_frames, -1, x.size(2), x.size(3))
        # t_fea_forward, _ = x.split([num_frames - 1, 1], dim=1)  # n, t-1, c//r, h, w
        # _, t_fea_backward = x.split([1, num_frames - 1], dim=1)  # n, t-1, c//r, h, w
        #
        # t_fea_backward = t_fea_backward - t_fea_forward
        # t_fea_backward = t_fea_backward.view((t_fea_backward.size(0), -1,)+t_fea_backward.size()[3:])

        x = self.fea_fusion(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        fea = x.view(x.size(0), -1)
        # print(x.size())

        # x = self.fc(fea)

        return fea

