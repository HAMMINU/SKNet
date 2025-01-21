import torch
from torch import nn

# thop 라이브러리에서 FLOPs(연산량)와 파라미터 수를 계산하는 함수 import
from thop import profile
from thop import clever_format

# SKConv 클래스 정의: Selective Kernel Convolution
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        """
        Args:
            features: 입력 채널 수 (number of input channels).
            M: 브랜치(branch)의 개수. 기본값은 2.
            G: 그룹 컨볼루션에서 그룹의 수.
            r: z 벡터의 길이를 계산할 때 사용하는 비율.
            stride: 스트라이드(Stride). 기본값은 1.
            L: z 벡터의 최소 길이. 기본값은 32.
        """
        super(SKConv, self).__init__()
        # z 벡터의 길이 계산. 최소값 L을 보장.
        d = max(int(features / r), L)
        self.M = M  # 브랜치 수
        self.features = features  # 입력 채널 수
        
        # 브랜치별로 서로 다른 커널 크기를 가진 컨볼루션 레이어 생성
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        
        # GAP(Global Average Pooling) 레이어
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # z 벡터를 생성하는 FC 레이어
        self.fc = nn.Sequential(
            nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        
        # 각 브랜치에 대한 attention vector를 계산하는 FC 레이어
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        
        # Softmax 함수로 attention weight 계산
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
            batch_size = x.shape[0]

            # 각 분기에서 특징 추출
            feats = [conv(x) for conv in self.convs]
            
            feats = torch.cat(feats, dim=1)  # 분기 결과를 채널 방향으로 연결
            # torch.cat을 통해 각 분기의 중요도 벡터를 채널 방향( dim = 1 )으로 결합합니다. 
            # ([8, 256, 56, 56])

            feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
            # 결합된 텐서를 [ 𝐵 , 𝑀 × 𝐶 , 1 , 1 ] 에서 [ 𝐵 , 𝑀 , 𝐶 , 1 , 1 ] 로 재구성합니다
            # ([8, 2, 128, 56, 56])  
            # ([8, [0], 128, 56, 56]) / ([8, [1], 128, 56, 56])
            # 배치,분기,입력채널,분기후 피쳐 크기    
            

            # 모든 분기의 특징을 합산하여 U 생성 
            feats_U = torch.sum(feats, dim=1)
            # ([8, 128, 56, 56])


            # Global Average Pooling으로 S 생성(Fgp)
            feats_S = self.gap(feats_U)
            # torch.Size([8, 128, 1, 1])

            # 채널 축소 및 활성화하여 Z 생성(Ffc)
            feats_Z = self.fc(feats_S)
            # torch.Size([8, 32, 1, 1])

            # 각 분기의 중요도 계산
            attention_vectors = [fc(feats_Z) for fc in self.fcs] 
            # A(cz), B(cz) : torch.Size([8, 128, 1, 1])
            attention_vectors = torch.cat(attention_vectors, dim=1)
            # AB : torch.Size([8, 256, 1, 1])
            attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
            # A / B : torch.Size([8, 2 , 128,  1, 1])
            attention_vectors = self.softmax(attention_vectors)
            # ac / bc : torch.Size([8, 2 , 128,  1, 1])

            # 중요도를 기반으로 각 분기의 특징 결합
            feats_V = torch.sum(feats * attention_vectors, dim=1)
            return feats_V
    
# SKNet의 기본 단위인 SKUnit 정의
class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """
        Args:
            in_features: 입력 채널 수.
            mid_features: 중간 채널 수.
            out_features: 출력 채널 수.
            M: 브랜치 수.
            G: 그룹 수.
            r: z 벡터의 길이를 계산할 때 사용하는 비율.
            stride: 스트라이드.
            L: z 벡터의 최소 길이.
        """
        super(SKUnit, self).__init__()
        
        # 첫 번째 1x1 컨볼루션 레이어
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
        )
        
        # SKConv 레이어
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        
        # 두 번째 1x1 컨볼루션 레이어
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
        )
        
        # Shortcut 설정
        if in_features == out_features:
            # 입력 차원과 출력 차원이 동일하면 그대로 사용
            self.shortcut = nn.Sequential()
        else:
            # 입력 차원과 출력 차원이 다르면 차원을 맞추기 위해 1x1 컨볼루션 추가
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x  # 입력을 저장 (shortcut에 사용)
        
        # 메인 경로
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        # Shortcut 연결과 합산
        return self.relu(out + self.shortcut(residual))


# SKNet 정의
class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list=[3, 4, 6, 3], strides_list=[1, 2, 2, 2]):
        """
        Args:
            class_num: 분류 클래스 개수.
            nums_block_list: 각 stage에서 사용할 블록 수.
            strides_list: 각 stage에서의 stride 값.
        """
        super(SKNet, self).__init__()
        # 기본 Conv 레이어
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Max Pooling 레이어
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 네 가지 stage를 정의
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected Layer
        self.classifier = nn.Linear(2048, class_num)
        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        # 첫 블록은 stride를 적용
        layers = [SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        # 나머지 블록은 stride 없이 추가
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.maxpool(fea)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


# SKNet 버전 정의
def SKNet26(nums_class=1000):
    return SKNet(nums_class, [2, 2, 2, 2])

def SKNet50(nums_class=1000):
    return SKNet(nums_class, [3, 4, 6, 3])

def SKNet101(nums_class=1000):
    return SKNet(nums_class, [3, 4, 23, 3])


# 메인 실행 코드
if __name__ == '__main__':
    # 입력 데이터 생성 (8개의 배치, 3채널, 224x224 이미지)
    x = torch.rand(8, 3, 224, 224)
    
    # SKNet26 모델 생성
    model = SKNet26()
    
    # 모델을 통해 출력 계산
    out = model(x)
    
    # FLOPs와 파라미터 수 계산
    flops, params = profile(model, (x, ))
    # 계산 결과를 사람이 읽기 쉽게 포맷팅
    flops, params = clever_format([flops, params], "%.5f")
    
    # 결과 출력
    print(flops, params)
    print('out shape : {}'.format(out.shape))
