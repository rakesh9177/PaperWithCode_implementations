import torch
import torch.nn as nn

class Adam:
    def __init__(self, model_parameters, loss_function, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        self.model_parameters = model_parameters
        self.m_t = [torch.zeros_like(param) for param in self.model_parameters]
        self.v_t = [torch.zeros_like(param) for param in self.model_parameters]
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self.loss_function = loss_function
    
    def step(self):
        self.t+=1
        self.m_t = [self.beta1 * m + (1 - self.beta1) * p.grad for p, m in zip(self.model_parameters, self.m_t)]
        self.v_t = [self.beta2 * v + (1 - self.beta2) * p.grad ** 2 for p, v in zip(self.model_parameters, self.v_t)]
        m_t_hat  = [m / (1 - self.beta1 ** self.t) for m in self.m_t]
        v_t_hat = [v / (1 - self.beta2 ** self.t) for v in self.v_t]
        for p, m, v in zip(self.model_parameters, m_t_hat, v_t_hat):
            p.data = p.data - self.lr * m / (v.sqrt() + self.epsilon)
    def zero_grad(self):
        for p in self.model_parameters:
            p.grad.zero_()
    def optimize(self, epochs = 10):
        loss = self.loss_function()
        for epoch in range(epochs):
            self.zero_grad()
            loss = self.loss_function()
            loss.backward()
            self.step()
        return self.model_parameters


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y = torch.tensor([[2.0], [4.0], [6.0]], requires_grad=False)


model = SimpleModel()
criterion = nn.MSELoss()

model_parameters = model.parameters()
adam = Adam(model_parameters=model_parameters, loss_function=lambda: criterion(model(x), y))
adam.optimize()
for param in model.parameters():
    print(param.data)




