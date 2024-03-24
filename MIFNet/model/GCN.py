import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, num_state, num_node):  
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):  
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()  
        seg_similar = torch.bmm(seg, aj)  
        out = self.relu(self.conv2(seg_similar))
        output = out + seg  
        return output


class EAGCN(nn.Module):
    def __init__(self, num_in=128, plane_mid=1, mids=(12,12), normalize=False): 
        super(EAGCN, self).__init__()
        self.num_in = num_in  
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)  
        self.num_n = (mids[0]) * (mids[1])
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids[0], mids[1]))

        self.gcn = GCN(num_state=num_in, num_node=self.num_n)  
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, seg_ori, edge_ori):
        seg = seg_ori
        edge = edge_ori 
        n, c, h, w = seg.size()  

   
        seg_s = self.conv_s1(seg) 
        theta_T = seg_s.view(n, self.num_s, -1).contiguous()  
        theta = seg_s.view(n, -1, self.num_s).contiguous()  
        channel_att = torch.relu(self.mlp(self.maxpool_c(seg).squeeze(3).squeeze(2))).view(n, self.num_s, -1) 
        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s)) 

        similarity_c = torch.bmm(theta, diag_channel_att)  
        similarity_c = self.softmax(torch.bmm(similarity_c, theta_T)) 

   
        seg_c = self.conv_s11(seg)  
        sigma = seg_c.view(n, self.num_s, -1).contiguous()  
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)  

        edge_m = seg * edge 

        maxpool_s, _ = torch.max(seg, dim=1)  
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        seg_ss = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)  
        edge_mm = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)  

        diag_spatial_att = torch.bmm(edge_mm, seg_ss) * sigma_out  
        similarity_s = self.softmax(diag_spatial_att) 
      
 

        similarity = similarity_c + similarity_s  

        seg_gcn = self.gcn(seg, similarity).view(n, self.num_in, self.mids[0], self.mids[1])  

        ext_up_seg_gcn = seg_gcn + seg_ori
        return ext_up_seg_gcn
    
    
if __name__ == '__main__':
    x = torch.randn(1 ,128 ,12 ,12)
    y = torch.randn(1 ,128 ,12 ,12)  
    net = EAGCN(128,1,(12,12))  
    out = net(x,y)
    print(out.shape)