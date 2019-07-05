

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv1 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv1 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv1 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.query_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query1  = self.query_conv1(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key1 =  self.key_conv1(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy1 =  torch.bmm(proj_query1,proj_key1) # transpose check
        attention1 = self.softmax(energy1) # BX (N) X (N) 
        proj_value1 = self.value_conv1(x).view(m_batchsize,-1,width*height) # B X C X N
        out1 = torch.bmm(proj_value1, attention1.permute(0,2,1) )
        out1 = out1.view(m_batchsize,C,width,height)

        proj_query2  = self.query_conv2(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key2 =  self.key_conv2(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy2 =  torch.bmm(proj_query2,proj_key2) # transpose check
        attention2 = self.softmax(energy2) # BX (N) X (N) 
        proj_value2 = self.value_conv2(x).view(m_batchsize,-1,width*height) # B X C X N
        out2 = torch.bmm(proj_value2,attention2.permute(0,2,1) )
        out2 = out2.view(m_batchsize,C,width,height)
        
        out = self.gamma1*out1 + self.gamma2*out2 + x
        return out, attention1, attention2