
from kan import *
def miningData():
    df=pd.read_csv('data/origindata.csv')
    print(df.head(5))
def createModel():
    name = input("Nhập tên của bạn: ")
    print(name)
    model = KANLayer(in_dim=3, out_dim=5)
    return model
print(torch.cuda.is_available())