import torch
import numpy as np
from torch import nn
from .mix_MLP import MLPMixer
import torch.nn.functional as F


class CascadeMixer(nn.Module):
    def __init__(self, input_nc=3, output_nc=3,  ngf=64, dim=256, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, n_downsampling=2):
        super(CascadeMixer, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        
        use_bias = nn.InstanceNorm2d
        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]

        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)

        # up_sample
        model_stream1_up = []
        model_stream2_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]
            model_stream2_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]                

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        model_stream2_up += [nn.ReflectionPad2d(3)]
        model_stream2_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream2_up += [nn.Tanh()]

        self.stream1_up = nn.Sequential(*model_stream1_up)
        self.stream2_up = nn.Sequential(*model_stream2_up)


        ######

        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream3 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream4 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream5 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream6 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream7 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream8 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream9 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream10 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        
        self.conv_block_stream11 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream12 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        
        self.conv_block_stream13 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream14 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream15 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream16 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)

        self.conv_block_stream17 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream18 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)


        self.mixer1 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer2 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer3 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer4 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer5 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer6 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv6 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer7 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=1)

        self.mixer8 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1)
        
        self.mixer9 = MLPMixer(in_chan=256, dim=128, patch_size=8, image_size=64, depth=7, token_dim=512, channel_dim=1024)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=1)


        self.bothFea = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)


    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2),
                       nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)



    def forward(self, x1, x2):
        # x1: segmentation[b, 3, 256, 256], x2: image [b, 3, 256, 256]

        # Step1: down_sample  outsize[B, 256, 64, 64]
        x1_out = self.stream1_down(x1)
        x2_out = self.stream2_down(x2)

        # Step2:Cascaded Mixer Blocks

        """----Block 1--"""

        ## Step2-1: Block1
        ### conv   out_size [B, 256, 64, 64]
        x1_block1 = self.conv_block_stream1(x1_out)
        x2_block1 = self.conv_block_stream1(x2_out)
        ### mixer MLP
        out_mix1 = self.mixer1(x1_block1, x2_block1)
        att1 = torch.sigmoid(out_mix1)
        ### update
        out2_block1 = x2_out + att1 * x2_block1

        out1_block1 = torch.cat((out_mix1, out2_block1), 1)
        out1_block1 = self.conv1(out1_block1)

        """----Block 2--"""
        ### conv out_size [B, 256, 64, 64]
        x1_block2 = self.conv_block_stream3(out1_block1)
        x2_block2 = self.conv_block_stream4(out2_block1)
        ### mixer  MLP
        out_mix2 = self.mixer2(x1_block2, x2_block2)
        att2 = torch.sigmoid(out_mix2)
        ### update
        out2_block2 = out2_block1 + att2 * x2_block2

        out1_block2 = torch.cat((out_mix2, out2_block2), 1)
        out1_block2 = self.conv2(out1_block2)

        """----Block 3--"""
        x1_block3 = self.conv_block_stream5(out1_block2)
        x2_block3 = self.conv_block_stream6(out2_block2)
        ### mixer  MLP
        out_mix3 = self.mixer3(x1_block3, x2_block3)
        att3 = torch.sigmoid(out_mix3)
        ### update
        out2_block3 = out2_block2 + att3 * x2_block3

        out1_block3 = torch.cat((out_mix3, out2_block3), 1)
        out1_block3 = self.conv3(out1_block3)

        """----Block 4--"""
        x1_block4 = self.conv_block_stream7(out1_block3)
        x2_block4 = self.conv_block_stream8(out2_block3)
        ### mixer  MLP
        out_mix4 = self.mixer4(x1_block4, x2_block4)
        att4 = torch.sigmoid(out_mix4)
        ### update
        out2_block4 = out2_block3 + att4 * x2_block4

        out1_block4 = torch.cat((out_mix4, out2_block4), 1)
        out1_block4 = self.conv4(out1_block4)

        """----Block 5--"""
        x1_block5 = self.conv_block_stream9(out1_block4)
        x2_block5 = self.conv_block_stream10(out2_block4)
        ### mixer  MLP
        out_mix5 = self.mixer5(x1_block5, x2_block5)
        att5 = torch.sigmoid(out_mix5)
        ### update
        out2_block5 = out2_block4 + att5 * x2_block5

        out1_block5 = torch.cat((out_mix5, out2_block5), 1)
        out1_block5 = self.conv5(out1_block5)

        """----Block 6--"""
        x1_block6 = self.conv_block_stream11(out1_block5)
        x2_block6 = self.conv_block_stream12(out2_block5)
        ### mixer  MLP
        out_mix6 = self.mixer6(x1_block6, x2_block6)
        att6 = torch.sigmoid(out_mix6)
        ### update
        out2_block6 = out2_block5 + att6 * x2_block6

        out1_block6 = torch.cat((out_mix6, out2_block6), 1)
        out1_block6 = self.conv6(out1_block6)

        """----Block 7--"""
        x1_block7 = self.conv_block_stream13(out1_block6)
        x2_block7 = self.conv_block_stream14(out2_block6)
        ### mixer  MLP
        out_mix7 = self.mixer7(x1_block7, x2_block7)
        att7 = torch.sigmoid(out_mix7)
        ### update
        out2_block7 = out2_block6 + att7 * x2_block7

        out1_block7 = torch.cat((out_mix7, out2_block7), 1)
        out1_block7 = self.conv7(out1_block7)

        """----Block 8--"""
        x1_block8 = self.conv_block_stream15(out1_block7)
        x2_block8 = self.conv_block_stream16(out2_block7)
        ### mixer  MLP
        out_mix8 = self.mixer8(x1_block8, x2_block8)
        att8 = torch.sigmoid(out_mix8)
        ### update
        out2_block8 = out2_block7 + att8 * x2_block8

        out1_block8 = torch.cat((out_mix8, out2_block8), 1)
        out1_block8 = self.conv8(out1_block8)

        """----Block 9--"""
        x1_block9 = self.conv_block_stream17(out1_block8)
        x2_block9 = self.conv_block_stream18(out2_block8)
        ### mixer  MLP
        out_mix9 = self.mixer9(x1_block9, x2_block9)
        att9 = torch.sigmoid(out_mix9)
        ### update
        out2_block9 = out2_block8 + att9 * x2_block9

        out1_block9 = torch.cat((out_mix9, out2_block9), 1)
        out1_block9 = self.conv9(out1_block9)


        # Step3: oganize output
        ### out segFeture  & imgFeature   [B, 256, 64, 64] -> [B, 128, 128, 128]
        bothFeature = self.bothFea(out2_block9)  
        ### out seg                       [B, 256, 64, 64] -> [B, 3, 256, 256]   
        out_seg = self.stream1_up(out1_block9)
        ### out Img                       [B, 256, 64, 64] -> [B, 3, 256, 256]
        out_img = self.stream2_up(out2_block9)

        return bothFeature, out_seg, out_img


if __name__ == "__main__":

    img1 = torch.ones([1, 3, 256, 256])
    img2 = torch.ones([1, 3, 256, 256])


    model = CascadeMixer(input_nc = 3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_downsampling=2)
   
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    outFea, outSeg, outImg = model(img1, img2)
    print("Shape of outFea :", outFea.shape)  # [B, in_channels, image_size, image_size]
    print("Shape of outSeg :", outSeg.shape)  # [B, in_channels, image_size, image_size]
    print("Shape of outImg :", outImg.shape)  # [B, in_channels, image_size, image_size]

