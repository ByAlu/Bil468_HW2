import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        # Hyperparameter will be added later
        self.network = nn.Sequential(
            
            nn.Linear(7560,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,1)
        )
    
    def forward(self,x):
        return self.network(x)
    
def main():
    model = MLP()

    print(model)
    print("Im here")

if __name__ == "__main__":
    main()